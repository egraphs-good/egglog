# Releasing egglog

An egglog release has two parts: publish the crates in this workspace, then
release `egglog-experimental` against the new egglog version. Publishing a
crate version is permanent, so run the dry run immediately before each real
publish and stop if it reports an error.

Set both versions before running the commands below. The experimental release
must use the same major version as the egglog release; its minor and patch
versions may differ. These example values do not prescribe the next release
number, and they do not include a leading `v`:

```sh
EGGLOG_RELEASE=3.0.0
EXPERIMENTAL_RELEASE=3.0.0
test "${EXPERIMENTAL_RELEASE%%.*}" = "${EGGLOG_RELEASE%%.*}"
```

## 1. Prepare the egglog release PR

Start from an up-to-date `main` and create a release branch:

```sh
git switch main
git pull --ff-only
git switch -c "release-v$EGGLOG_RELEASE"
```

Make these edits:

1. In `Cargo.toml`, update `workspace.package.version` and the `version` of
   every internal crate in `workspace.dependencies`. All publishable workspace
   crates use the same release version, and the dependency versions must match
   it.
2. In `README.md`, update the example egglog dependency version.
3. In `CHANGELOG.md`, rename the current `Unreleased` section to the new
   version and release date, add a new empty `Unreleased` section, update its
   comparison link, and add a link for the new version. Do not rewrite links
   for old releases. Check every entry against the git history, put it under
   the release that first contained the change, and include the relevant PR
   number or numbers.
4. Search for any remaining references to the old version and decide whether
   each should change:

   ```sh
   rg '2\.0\.0|v2\.0\.0'
   ```

   Replace `2.0.0` in that command with the version being superseded.

Regenerate the workspace entries in `Cargo.lock`, then run the release checks:

```sh
cargo update --workspace
make all
git diff --check
git status --short
```

Commit the changes. From that clean commit, verify all of the publishable
packages as a set:

```sh
cargo publish --dry-run --workspace --exclude egglog-wasm-example
```

This catches packaging errors across the workspace before any upload is made.
Then push the branch and open a PR. Let the normal review process finish: wait
for CI and human approval, and do not have the agent that prepared the PR merge
it. A maintainer should merge it after review.

## 2. Tag the egglog release

Tag the merged commit, not the pre-merge release-branch commit:

```sh
git switch main
git pull --ff-only
git status --short
git tag -a "v$EGGLOG_RELEASE" -m "egglog $EGGLOG_RELEASE"
git push origin "v$EGGLOG_RELEASE"
```

`git status --short` should print nothing. Check that the tag points at the
intended release commit before publishing:

```sh
git show --stat "v$EGGLOG_RELEASE"
```

## 3. Publish the egglog workspace

Authenticate once with `cargo login` if this machine does not already have a
crates.io token.

The workspace crates must be published in dependency order. The crate list can
change, so regenerate this report for every release (it requires `jq`):

```sh
cargo metadata --format-version 1 |
  jq -r '
    . as $m
    | $m.packages[]
    | select(.id as $id | $m.workspace_members | index($id))
    | select(.publish != [])
    | [.name, ([.dependencies[]
        | select(.path != null and .kind != "dev")
        | .name] | unique | join(", "))]
    | @tsv
  ' | sort
```

The first column lists every publishable workspace crate; the second lists its
non-development workspace dependencies. Compare the report with the command
block below: every crate must appear exactly once, and every dependency must be
published before the crate that depends on it. Update the block whenever a
crate or dependency is added. Then run each dry-run and publish pair
separately, from the repository root, and do not proceed to the next pair
unless both commands succeed:

```sh
cargo publish --dry-run -p egglog-numeric-id
cargo publish -p egglog-numeric-id

cargo publish --dry-run -p egglog-reports
cargo publish -p egglog-reports

cargo publish --dry-run -p egglog-ast
cargo publish -p egglog-ast

cargo publish --dry-run -p egglog-add-primitive
cargo publish -p egglog-add-primitive

cargo publish --dry-run -p egglog-concurrency
cargo publish -p egglog-concurrency

cargo publish --dry-run -p egglog-union-find
cargo publish -p egglog-union-find

cargo publish --dry-run -p egglog-core-relations
cargo publish -p egglog-core-relations

cargo publish --dry-run -p egglog-bridge
cargo publish -p egglog-bridge

cargo publish --dry-run -p egglog
cargo publish -p egglog
```

`egglog-wasm-example` has `publish = false` and is intentionally omitted. The
four crates without internal dependencies at the start can be published in any
order, but the order shown above is a valid order for the entire workspace.

crates.io index updates can take a little time. If the next crate reports that
the version it depends on is unavailable, wait and retry that next crate's dry
run. If `cargo publish` times out while polling the index, first check whether
the upload succeeded before running it again:

```sh
cargo info "egglog-numeric-id@$EGGLOG_RELEASE"
```

Substitute the crate that was just uploaded. Publishing many crates can also
trigger crates.io rate limiting. If that happens, do not retry in a tight loop:
wait for the cooldown, verify whether the current upload succeeded with
`cargo info`, and resume with that crate's dry run. After the final publish,
verify the main package:

```sh
cargo info "egglog@$EGGLOG_RELEASE"
```

## 4. Release egglog-experimental

Do this only after the new egglog packages are visible on crates.io. In a
separate clone of <https://github.com/egraphs-good/egglog-experimental>:

```sh
git switch main
git pull --ff-only
git switch -c "release-v$EXPERIMENTAL_RELEASE"
```

Update its `Cargo.toml` as follows:

1. Set `package.version` to `$EXPERIMENTAL_RELEASE`.
2. Search every `Cargo.toml` for `egglog` and update every dependency package
   whose name starts with `egglog`, including renamed, development, build, and
   target-specific dependencies:

   ```sh
   rg -n 'egglog' --glob Cargo.toml
   ```

   For crates.io, each dependency must have a `version` whose value is
   `$EGGLOG_RELEASE`. It may also retain a `git` and `rev` for repository
   development; if so, point `rev` at the tagged egglog release commit. Cargo
   uses the Git source locally and the versioned registry source in the
   published package.
3. In `CHANGELOG.md`, move the user-facing changes from `Unreleased` into a
   section for `$EXPERIMENTAL_RELEASE` and the release date, add a new empty
   `Unreleased` section, and update the comparison links.
4. Update the README dependency example.
5. Review the crate, module, and public-item rustdoc so it accurately describes
   the released features and APIs.

Update its lockfile and test it:

```sh
cargo update
cargo test --release
cargo fmt --check
cargo clippy --tests -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
git diff --check
git status --short
```

Commit these changes, run `cargo publish --dry-run` from the clean commit, and
open an egglog-experimental release PR. Wait for CI and human approval, and do
not have the agent that prepared the PR merge it. A maintainer should merge it
after review. Then tag the merged commit and publish it:

```sh
git switch main
git pull --ff-only
git status --short
git tag -a "v$EXPERIMENTAL_RELEASE" -m "egglog-experimental $EXPERIMENTAL_RELEASE"
git push origin "v$EXPERIMENTAL_RELEASE"

cargo publish --dry-run
cargo publish
cargo info "egglog-experimental@$EXPERIMENTAL_RELEASE"
```

The tags and crates.io uploads complete the release. These repositories
currently use tags only; do not create GitHub Release objects.

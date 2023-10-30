

How to do a release:
1. Update `CHANGELOG.md` with a new entry and new link at the bottom.
2. Find and replace in the codebase to update the version number. Make sure to get `Cargo.toml` and places in the changelog. Be careful not the screw up old links though!
4. Commit.
5. Tag the commit with the version number.
6. Make a PR and make sure the tag gets added.
7. Merge the PR
8. `cargo publish --dry-run`
   1. Sometimes this can result in an error- you may need to run `cargo update` to update `cargo.lock`
9.  `cargo publish`

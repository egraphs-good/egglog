

How to do a release:
1. Update `CHANGELOG.md` with a new entry and new link at the bottom.
2. Find and replace in the codebase to update the version number. Make sure to get `Cargo.toml` and places in the changelog. Be careful not the screw up old links though!
3. Commit.
4. Tag the commit with the version number.
5. Make a PR and make sure the tag gets added.
6. Merge the PR
7. `cargo publish --dry-run`
8. `cargo publish`

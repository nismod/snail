---
name: Release snail
about: Checklist to follow for release
title: "(release) v0."
labels: ''
assignees: ''

---

Follow these steps:
- [ ] Merge all desired PRs to `main`, let CI tests pass
- [ ] Update version number in `pyproject.toml`
- [ ] Push to `main`
- [ ] Start drafting a new release at [https://github.com/nismod/snail/releases/new](https://github.com/nismod/snail/releases/new)
  - [ ] Create a new tag: click "Choose a tag", enter new version number (e.g. `v0.5.2`), click "Create new tag on publish"
  - [ ] Click "Generate release notes" (this should give the release the same title as the tag (e.g. "v0.5.2")
  - [ ] Edit the "What's New" section into "Features" and "Fixes", simplify and combine PR bullet points
  - [ ] Publish release
- [ ] Wait for CI to run - this should trigger the "package" workflow in GitHub Actions and push to [PyPI](https://pypi.org/project/nismod-snail/#history)

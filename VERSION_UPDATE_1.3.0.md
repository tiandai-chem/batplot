# Version Update Summary: v1.3.0

## Date: October 21, 2025

## Version Changes

### Previous Version
- **1.2.1** (January 16, 2025)

### New Version
- **1.3.0** (October 21, 2025)

---

## Files Updated

### 1. pyproject.toml
- **Line 7**: `version = "1.2.1"` → `version = "1.3.0"`
- This is the primary version source for package distribution

### 2. batplot/__init__.py
- **Line 4**: `__version__ = "1.0.18"` → `__version__ = "1.3.0"`
- Python package version identifier
- Used for runtime version checks

### 3. USER_MANUAL.md
- **Line 2**: `v1.1.1, 2025-01-16` → `v1.3.0, 2025-10-21`
- User-facing documentation version
- Updated date to release date

### 4. CHANGELOG.md (NEW)
- Created comprehensive changelog documenting all changes
- Follows [Keep a Changelog](https://keepachangelog.com/) format
- Adheres to [Semantic Versioning](https://semver.org/)

---

## Why Version 1.3.0?

### Semantic Versioning Breakdown

**Format**: MAJOR.MINOR.PATCH

- **MAJOR (1)**: No breaking changes, kept at 1
- **MINOR (2→3)**: New features added in backward compatible manner
- **PATCH (1→0)**: Reset to 0 for new minor version

### New Features Justifying Minor Version Bump

1. **Batch Style Support** (Major Feature)
   - New `--all` flag with style file argument
   - New functions: `_load_style_file()`, `_apply_ec_style()`
   - Applies to all EC modes (GC, CV, dQdV, CPC)
   - Backward compatible (old syntax still works)

2. **Tick Length Control** (Feature Addition)
   - New `l` command in all interactive menus
   - Session persistence
   - Undo/redo integration
   - Style export/import support

3. **Tick Direction Control** (Feature Addition)
   - New `d` command in toggle menu
   - Full integration with state management
   - Session and style persistence

4. **CIF Title Toggle** (Feature Addition)
   - New `h` command for XY plots
   - Session persistence
   - Undo/redo support

All features are **additions** (not breaking changes), making this a **minor version** increment.

---

## Version Consistency Check

✅ **pyproject.toml**: 1.3.0  
✅ **batplot/__init__.py**: 1.3.0  
✅ **USER_MANUAL.md**: v1.3.0  
✅ **CHANGELOG.md**: [1.3.0] documented  

All version identifiers are now synchronized.

---

## Release Checklist

### Pre-Release
- [x] Version numbers updated in all files
- [x] CHANGELOG.md created and populated
- [x] Documentation updated (USER_MANUAL.md, README.md)
- [x] Help text updated (args.py)
- [x] All new features documented
- [x] Code compiles without errors

### Testing (Before Publishing)
- [ ] Test batch style with .bps files
- [ ] Test batch style with .bpsg files
- [ ] Test all EC modes (GC, CV, dQdV, CPC)
- [ ] Test tick length adjustment
- [ ] Test tick direction toggle
- [ ] Test CIF title toggle
- [ ] Test undo/redo for all new features
- [ ] Test session save/load
- [ ] Test style export/import

### Building
- [ ] Clean old builds: `rm -rf dist/ build/ *.egg-info`
- [ ] Build package: `python -m build`
- [ ] Verify build artifacts in `dist/`

### TestPyPI (Recommended First)
- [ ] Upload: `python -m twine upload --repository testpypi dist/*`
- [ ] Install: `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple batplot==1.3.0`
- [ ] Test installation
- [ ] Verify all features work

### PyPI (Production)
- [ ] Upload: `python -m twine upload dist/*`
- [ ] Verify on PyPI: https://pypi.org/project/batplot/
- [ ] Test installation: `pip install batplot==1.3.0`
- [ ] Update GitHub repository

### Post-Release
- [ ] Tag release in git: `git tag v1.3.0`
- [ ] Push tags: `git push origin v1.3.0`
- [ ] Create GitHub release with CHANGELOG notes
- [ ] Announce release (if applicable)

---

## What's New in v1.3.0

### User-Facing Features

**1. Batch Styling** 🎨
Apply consistent formatting to all your EC plots with one command:
```bash
batplot --all mystyle.bps --gc --mass 7.0
```

**2. Tick Length Control** 📏
Adjust tick sizes interactively (press `t` → `l`):
- Set major tick length
- Minor ticks auto-calculated (70% of major)
- Applies to all axes

**3. Tick Direction** ↔️
Toggle tick direction (press `t` → `d`):
- Options: in, out, inout
- Applies to all ticks

**4. CIF Title Toggle** 🏷️
Hide/show CIF peak labels (press `h`):
- Cleaner plots when needed
- Toggle on/off anytime

### Under the Hood

- Full undo/redo integration for all features
- Session persistence (everything saved in .pkl files)
- Style export/import support
- Comprehensive documentation
- Backward compatible (no breaking changes)

---

## Migration Guide

### From 1.2.1 to 1.3.0

**Good News**: No migration needed! All existing commands work as before.

**New Capabilities**:
```bash
# Old way (still works)
batplot --gc all --mass 7.0

# New way (with styling)
batplot --all mystyle.bps --gc --mass 7.0
```

**Session Files**: Old .pkl files are fully compatible. New features will have default values.

**Style Files**: Old .bps/.bpsg files work. New files include tick parameters.

---

## Breaking Changes

**None!** Version 1.3.0 is fully backward compatible with 1.2.1.

---

## Next Steps for Publishing

1. **Test thoroughly** with real data files
2. **Build package**: `python -m build`
3. **Upload to TestPyPI** first
4. **Test installation** from TestPyPI
5. **Upload to PyPI** when confident
6. **Tag release** in git
7. **Update GitHub** with release notes

---

## Notes

- This is a **feature release**, not a bugfix release
- All new features are **optional** and don't affect existing workflows
- Documentation is comprehensive and up-to-date
- Code quality maintained (no syntax errors)

---

**Version**: 1.3.0  
**Release Date**: October 21, 2025  
**Status**: Ready for Testing → Publishing  
**Backward Compatible**: Yes ✅

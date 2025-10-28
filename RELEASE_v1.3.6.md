# Release Summary: batplot v1.3.6

## Release Information
- **Version**: 1.3.6
- **Release Date**: October 27, 2025
- **PyPI Link**: https://pypi.org/project/batplot/1.3.6/

## Installation
```bash
pip install --upgrade batplot
```

## What's New in v1.3.6

### 🎯 Key Feature: Optional Current Column for Operando EC
- `.mpt` files with only time and voltage columns can now be used for operando EC plots
- Current column is no longer required for basic operando electrochemistry visualization
- Advanced features (ion counting) properly handle missing current data with clear error messages

### 📧 Mailing List Announcement
Users are now encouraged to subscribe to **batplot-lab@kjemi.uio.no** for:
- Feature updates and release announcements
- Community discussions and feedback
- Tips and best practices

### 🔧 Technical Improvements
- Enhanced error handling for missing data columns
- Better user feedback for feature requirements
- Improved code robustness for partial data files

## Files Updated in This Release

### Core Code Changes
- `batplot/readers.py` - Made current column optional in `read_mpt_file()`
- `batplot/operando_ec_interactive.py` - Added validation for ion counting features

### Version Updates
- `pyproject.toml` - Updated version to 1.3.6
- `batplot/__init__.py` - Updated version to 1.3.6

### Documentation Updates
- `README.md` - Added mailing list subscription information
- `USER_MANUAL.md` - Added mailing list to contact section
- `batplot/args.py` - Added mailing list to help messages
- `CHANGELOG_v1.3.6.md` - Complete changelog for this release

## Migration Guide
No migration needed! This release is fully backward compatible:
- ✅ Existing `.mpt` files with all columns work as before
- ✅ New `.mpt` files with only time+voltage now supported
- ✅ All existing functionality preserved

## Testing Recommendations
1. Test operando plotting with `.mpt` file containing only time and voltage
2. Verify ion counting displays appropriate error without current data
3. Confirm full functionality works with complete `.mpt` files

## Distribution Files
- `batplot-1.3.6-py3-none-any.whl` (204.2 KB)
- `batplot-1.3.6.tar.gz` (195.3 KB)

## Build Artifacts Cleaned
- Removed old `dist/`, `build/`, and `batplot.egg-info/` directories
- Fresh build for v1.3.6

## Next Steps for Users
1. **Upgrade**: `pip install --upgrade batplot`
2. **Subscribe**: Join batplot-lab@kjemi.uio.no for updates
3. **Explore**: Try the new optional current column feature
4. **Feedback**: Share your experience with the community

## Links
- **PyPI**: https://pypi.org/project/batplot/1.3.6/
- **GitHub**: https://github.com/tiandai-chem/batplot
- **Mailing List**: batplot-lab@kjemi.uio.no
- **Author**: Tian Dai (tianda@uio.no)

---

**Published successfully to PyPI on October 27, 2025** 🎉

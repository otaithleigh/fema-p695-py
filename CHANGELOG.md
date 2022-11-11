Changelog
=========

[Upcoming]
----------

### Added

- `CHANGELOG.md`

### Changed

- Warnings raised by `seismic_response_coeff` now appear as coming from the
  calling function.

### Fixed

- Limitation on period passed to `snrt` is now correctly $0.25 \le T \le 5.0$,
  instead of $0.25 \lt T \lt 5.0$.


[0.2.1] - 2022-05-12
--------------------

### Fixed

- Fix spectral shape factor calculation for Dmin
  - The text of Appendix B states that $\varepsilon_o$ = 1.5 for "SDC D", with
    no mention of Dmax vs. Dmin. But the tabulated SSF in both Ch. 7 and App. B
    show the SSF for Dmin as being the same as SDC B and C, which corresponds to
    an $\varepsilon_o$ of 1.0. For consistency with the tabulated values this
    package now uses 1.0 for Dmin.
- Suppress `h5py` warning about change to default file mode by explicitly
  specifying mode 'r'


[0.2.0] - 2022-04-21
--------------------

### Added

- New function `snrt` to calculate $\hat{S}_{\mathit{NRT}}$ for either the far-field or near-field record set.
- `sf1` now takes an optional argument `record_set` to select between farfield
  and nearfield. Defaults to `'farfield'` to keep compatibility with previous
  behavior which was limited to the far-field set.
- Docstrings have been updated and normalized throughout.

### Fixed

- $\hat{S}_{\mathit{NRT}}$ is now calculated using the standard tabulated values
  from FEMA P695.


[0.1.0] - 2021-10-21
--------------------

Initial release.

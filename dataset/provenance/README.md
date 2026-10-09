# Provenance artefacts

One file per **source page**, holding a saved copy of the licence statement as it
appeared when we downloaded the audio (spec 4.1, rule 4). `scripts/check_provenance.py`
fails the build if one is missing.

## What to save

For each entry in `../SOURCES.yaml`, open its `page_url` and save the page — or at
minimum the part showing the licence / copyright / rights field — under the exact
filename given in that entry's `provenance:` field:

| File | Page to capture |
|---|---|
| `S1_librivox_gettysburg_150th.html` ✅ | LibriVox *Gettysburg Address 150th Anniversary* catalogue page |
| `S1b_librivox_gettysburg_v2.html` ✅ | LibriVox *Gettysburg Address (version 2)* page |
| `S2_jfk_library_jfkwha001.pdf` ⬜ | JFK Library JFKWHA-001 asset page, showing **Copyright Status** |
| `S3_commons_roosevelt_pearl_harbor.html` ✅ | Wikimedia Commons file page, showing the licence box |
| `S4_librivox_civil_war_vol1_ch1.pdf` ⬜ | LibriVox *American Civil War Collection Vol. 1* page (optional source) |

PNG or JPG is accepted in place of PDF — update the `provenance:` field in
`SOURCES.yaml` to match the extension you actually saved.

## How

Browser → Print → *Save as PDF* keeps the URL and the date in the header, which is
what makes the artefact worth anything. A cropped screenshot with no URL visible
does not.


## Status (2026-10-04)

✅ = captured, ⬜ = still to do by a human.

The three ✅ files are **raw HTML captures** of the page, fetched with `curl` on
2026-10-04 and saved verbatim; see `CAPTURE_LOG.md` for the URL, HTTP status,
byte count and SHA-256 of each. They are machine-saved evidence, not browser
print-to-PDF, so they do not carry a rendered URL/date header. The licence
sentence is present in the saved bytes and is quoted in `SOURCES.yaml`. If the
submission wants a human-readable artefact, print the page to PDF as described
above and point `provenance:` at the PDF instead — the checker only requires
the named file to exist.

S2 (JFK Library) could not be captured from this machine: the site returns
HTTP 403 to every automated request, including `robots.txt`. It needs a real
browser.

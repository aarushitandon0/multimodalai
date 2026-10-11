# Provenance capture log

Raw page captures saved by `curl -L` on 2026-10-04 (UTC), and S5-S7 on 2026-10-11, from this repository's
working copy. Each returned HTTP 200. The licence sentence quoted in
`../SOURCES.yaml` for that source appears verbatim in these bytes.

| File | Source page | Bytes | SHA-256 |
|---|---|---|---|
| `S1_librivox_gettysburg_150th.html` | <https://librivox.org/the-gettysburg-address-150th-anniversary-by-abraham-lincoln/> | 28620 | `0575961887ba9f4271cdcda90ed26ff1f6ef2dd63775e10f19b782f422750293` |
| `S1b_librivox_gettysburg_v2.html` | <https://librivox.org/the-gettysburg-address-by-abraham-lincoln-version-2/> | 20501 | `7a5cb92b48d07d0d193d0242f62e5598c3ed03e67703af55e87d18704cc26e10` |
| `S3_commons_roosevelt_pearl_harbor.html` | <https://commons.wikimedia.org/wiki/File:Roosevelt_Pearl_Harbor.ogg> | 147505 | `967246a8fc7cb43fbfe04a14d8829458f3753c0690b7ea8b66edc0be2c71cfd0` |
| `S5_librivox_im_nobody.html` | <https://librivox.org/im-nobody-by-emily-dickinson/> | 27907 | `4234666555fe1a97eb450c4d90abfdcf1ffd867d2bf9a7e9bc2836cf1e791004` |
| `S6_librivox_sunset_lampman.html` | <https://librivox.org/sunset-by-archibald-lampman/> | 26484 | `aeefc25201adcd0ce34ab0ec5098cfe7276d16cd2c6defcc96cf42c0de303cf3` |
| `S7_librivox_sonnet18.html` | <https://librivox.org/sonnet-18-by-william-shakespeare-2/> | 27093 | `21d73a629d32ee5cfa4a2762b95d45cdf9cf925bf1a9e364b4be61fe7c31ed77` |

## Not captured

| File | Source page | Why |
|---|---|---|
| `S2_jfk_library_jfkwha001.pdf` | <https://www.jfklibrary.org/asset-viewer/archives/jfkwha-001> | Site returns HTTP 403 to automated requests (incl. `robots.txt`); needs a real browser. |
| `S4_librivox_civil_war_vol1_ch1.pdf` | not yet identified | Optional source; its LibriVox page URL is still `TODO(human)`. |

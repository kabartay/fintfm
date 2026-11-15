# UK Companies House as the forward-register data source (task 6.1)

**Date checked: 2026-11-15.** Every claim below was read on the page named beside it on that
date. The first pass went through WebFetch, which returns a summary, so the quotes were
re-checked the same day against the raw page text (HTML fetched with `curl`, tags stripped, each
quoted sentence searched verbatim; PDF text via `pdftotext`). The result is in "Verification
(2026-11-15, raw page text)" at the end: every quote below is verbatim except where that
section records a correction, and those corrections are applied in the text. Nothing was
downloaded beyond product pages, one OpenAPI document and one PDF; no account or API key was
created, so nothing here comes from a live API call. Status of every number: READ FROM A PAGE,
not measured by us.

## Gaps first

- **No formal licence is stated on any bulk product page.** The free company data, accounts
  data and PSC pages carry no licence, copyright or terms section; their only terms-type
  sentence is that the product is "provided free of charge" and "will not be supported" (basic
  company data, PSC) or "is not supported" (accounts data). The Open Government Licence is
  stated for Companies House's own publications and for The Gazette, but **not** for the register
  data. What Companies House does say about register data is that it "imposes no rules or
  requirements on how the information on the public register is used". That is permission by
  absence of conditions, not a licence grant. See "Licence" below and the "not established" list.
- **The monthly basic-data snapshot covers live companies only**, so companies that have
  dissolved drop out of it. Outcomes therefore cannot be read from later snapshots alone.
- **The API needs a registered account and key**, which this task did not create.
- **The size of the cohort that has accounts in structured form is not known.** The accounts
  product only holds electronically filed accounts.

## Products

| product | what it is | update frequency | format | source |
| --- | --- | --- | --- | --- |
| Free Company Data Product (basic company data) | "a downloadable data snapshot" "containing basic company data of live companies on the register" | "updated within 5 working days of the previous month end" | CSV in zip; one 471Mb file or seven parts (page shows 2026-11-01 files) | https://download.companieshouse.gov.uk/en_output.html |
| Accounts Data Product | "a free downloadable ZIP file, which contains the individual data files" | "A new daily file will be added each morning over a 5-day period (Tuesday-Saturday)"; each daily file "will only be available to download for 60 calendar days"; "historic accounts data is available from the previous year's monthly files" | individual files are inline XBRL (.html), XBRL (.xml) or a zip of iXBRL; "Data is only available for electronically filed accounts"; file names carry the company number | https://download.companieshouse.gov.uk/en_accountsdata.html |
| PSC snapshot | "a downloadable data snapshot containing the full list of PSC's provided to Companies House" | "updated every morning before 10am GMT"; "All files are overwritten daily" | JSON, 33 parts (page shows 2026-11-15 files) | https://download.companieshouse.gov.uk/en_pscdata.html |

Consequences for the register:

- A monthly basic-data snapshot taken on a fixed date gives a mechanical, hashable universe
  (task 6.3). The page lists dated file names, so a snapshot can be archived by us on the day
  it is fetched; whether Companies House keeps older monthly snapshots online was **not**
  established (the page shows only the current month).
- The accounts files are retained as daily files for 60 days and monthly files for the previous
  year, so the filings behind a cohort must be **archived by us when fetched**; they cannot be
  re-fetched later on the strength of this page.

## Fields in the basic-data snapshot

From the data-fields PDF linked on the product page
(https://resources.companieshouse.gov.uk/toolsToHelp/pdf/freeDataProductDataset.pdf, PDF created
15 May 2012 per its metadata; text read with `pdftotext`). Field names as printed:

- Identity: CompanyName, CompanyNumber, registered office address fields, CompanyCategory,
  CompanyStatus, CountryofOrigin.
- Dates: **Dissolution Date**, IncorporationDate.
- Accounts: AccountingRefDay, AccountingRefMonth, NextDueDate, LastMadeUpDate, AccountsCategory.
- Returns, and confirmation statement due and last-made-up dates.
- Mortgages: NumMortCharges, NumMortOutstanding, NumMortPartSatisfied, NumMortSatisfied.
- SIC codes (up to four), limited-partnership partner counts, previous names (up to ten).

It carries **no financial statement values**. Financials come only from the accounts product
(iXBRL). The PDF is dated 2012 and the snapshot may have gained fields since: the PDF was not
checked against a downloaded file (no bulk download was allowed).

## Identifying outcomes and their dates

Primary route (API, key needed): the Companies House Public Data API, specification at
https://developer-specs.company-information.service.gov.uk/companies-house-public-data-api/resources/companyprofile?v=latest
and `.../resources/companyinsolvency?v=latest`, read in full.

- `GET https://api.company-information.service.gov.uk/company/{companyNumber}` (company profile,
  source: the reference page
  https://developer-specs.company-information.service.gov.uk/companies-house-public-data-api/reference/company-profile/company-profile).
  `company_status` possible values: active, dissolved, liquidation, receivership, administration,
  voluntary-arrangement, converted-closed, insolvency-proceedings, registered, removed, closed,
  open. `company_status_detail` includes `active-proposal-to-strike-off`.
  The profile schema has `date_of_cessation` and `date_of_creation`; the descriptions for those
  two were not captured. `has_been_liquidated` and `has_insolvency_history` are marked
  "Deprecated. Please use links.insolvency".
- Insolvency resource: `status` values include liquidation, `in-administration`,
  administration-order, receivership, voluntary-arrangement.
  `cases[].type` values: compulsory-liquidation, creditors-voluntary-liquidation,
  members-voluntary-liquidation, in-administration, corporate-voluntary-arrangement, receivership
  and others. `cases[].dates[].type` values include petitioned-on, ordered-to-wind-up-on,
  wound-up-on, administration-started-on, due-to-be-dissolved-on, case-end-on,
  concluded-winding-up-on.
- Search has a dedicated "search dissolved companies" method (listed in the same API reference
  navigation).
- Bulk route, no key: CompanyStatus and Dissolution Date columns of a monthly snapshot. Because
  the snapshot lists live companies, a dissolved company is expected to be absent from later
  snapshots rather than shown as dissolved. **Not established** whether recently dissolved
  companies stay in a snapshot for a period.

Second route, no Companies House account: The Gazette. Its data page
(https://www.thegazette.co.uk/data) lists company notices since 1998 including "insolvency",
and a notice search (https://www.thegazette.co.uk/all-notices/notice) offers the corporate types
"Administration", "Creditors' voluntary liquidation", "Liquidation by the Court", "Members'
voluntary liquidation", "Moratoria", "Receivership" and others. Gazette notices are the
independent public record for the cross-check in 6.5; the machine-readable formats the data page
lists are PDF, TTL, RDF, JSON LD, HTML and XML via a linked data API and a SPARQL endpoint.

**Which events count as the outcome is a design decision for 6.3, not settled here.**
Members' voluntary liquidation is a solvent wind-up and is not a default; dissolution after a
strike-off can be dormancy rather than distress. The enumerations above make the distinction
possible (compulsory and creditors' voluntary liquidation, administration, receivership against
members' voluntary liquidation and strike-off).

## Rate limits

- Companies House API: "You can make up to 600 requests within a five-minute period." Exceeding
  it returns "429 Too Many Requests". Source:
  https://developer-specs.company-information.service.gov.uk/guides/rateLimiting
- Companies House API, developer guidelines (https://developer.company-information.service.gov.uk/developer-guidelines):
  the same 600 requests per 5 minutes; "If you have an application that requires a higher rate
  limit than this default, contact us." and "We reserve the right to ban without notice
  applications that regularly exceed or attempt to bypass the rate limits."
- Gazette fair use policy (https://www.thegazette.co.uk/fair-use-policy), read: for a bot, crawler,
  scraper or other automated tool that accesses the website or APIs, "Carry out crawling in
  non-business hours between 9pm and 7am UK." and "Limit requests to 5 request every 10 seconds."
  (the page's own wording); also a unique User-Agent with a contact method, and "Contact us for
  high-volume use". Arithmetic (ours): 5 requests per 10 seconds is 1,800 per hour, so a
  10-hour night window allows 18,000.
- Companies House searchable databases (website search) are "not intended to be a source for bulk
  downloads" (https://www.gov.uk/guidance/searching-the-companies-house-register); bulk data
  comes from the data products, not by crawling the website.
- Arithmetic (ours, not a source figure): 600 requests per 5 minutes is 7,200 per hour. A status
  check of 100,000 firms at one request each takes about 14 hours of continuous calling, and
  fetching the insolvency resource as well doubles it. Use the bulk snapshot for the universe and
  the API only for the firms whose status changed or went missing.

## Licence

| source | stated terms | page |
| --- | --- | --- |
| Companies House publications | "Material of this sort is subject to Crown copyright protection" and "is made available on GOV.UK under the Open Government Licence v3.0, except where otherwise stated." | https://www.gov.uk/government/publications/companies-house-accreditation-to-information-fair-traders-scheme/public-task-copyright-and-crown-copyright |
| Public register information | "Companies House imposes no rules or requirements on how the information on the public register is used." Made available "by virtue of approvals issued by us in accordance with section 47 of the Copyright, Designs and Patents Act 1988 and Schedule 1 of the Database Regulations (SI 1997/3032)." | same page, section 2.2 |
| Re-use | "We place no restriction on how the information is used after purchase, other than the following: all customers, including bulk customers, must take their own legal advice regarding possible breach of third party copyright; customers cannot reproduce the Crown insignia; if information is used from guidance notes, the website, publications or statistical tables the customer must credit Companies House as the source of the information" | same page, section 3 |
| Developer forum, Service Owner for Get Company Information, 4 Nov 2021 | "Companies House imposes no rules or requirements on how the information on the public register is used." "Companies House is not responsible for your use of the company data." "You are responsible for complying with any applicable data protection, copyright and other legislation and regulations." The reply does not mention the OGL by name and does not confirm that it applies to API data. | https://forum.companieshouse.gov.uk/t/is-data-provided-under-the-ocs/4513 |
| The Gazette | "All content is available under the Open Government Licence v3.0, except where otherwise stated." "please note that this licence does not cover the re-use of personal data." | https://www.thegazette.co.uk/data |
| Open Government Licence v3.0 | "exploit the Information commercially and non-commercially for example, by combining it with other Information, or by including it in your own product or application." Condition: "acknowledge the source of the Information in your product or application" using "Contains public sector information licensed under the Open Government Licence v3.0." Not covered: "personal data in the Information", "third party rights the Information Provider is not authorised to license", logos and crests, patents, trade marks and design rights. Grant: "a worldwide, royalty-free, perpetual, non-exclusive licence". | https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ |

Reading, kept to what the pages say:

- **Commercial use of register data is not restricted by Companies House**, on its own statement,
  and the stated restrictions are third-party copyright, the Crown insignia, and personal data
  law. That is a statement of no conditions, not an OGL grant. Do not write "OGL v3.0" against
  the register data or the bulk products in any public document: no page read states it. A 2014
  FOI answer reported by a search summary said the free product was not offered under the OGL;
  that source was **not opened** and is a lead only.
- **The Gazette is explicitly OGL v3.0**, with attribution and a personal-data carve-out, and
  that explicitly includes commercial use.
- **Third-party copyright** is the live risk named by Companies House itself. The accounts
  product reproduces filed accounts prepared by the companies and their accountants. Whether
  that raises any practical constraint on deriving numeric features from them is **not
  established** and is a question for the owner and a lawyer, not for this note.

## Personal data

- Basic company data: registered office address, and a "Careof" field, are in the file; no
  officer or individual fields are listed in the data-fields PDF. Company registered offices may
  still be an individual's home address; not established how often.
- PSC snapshot: lists people with significant control, so it holds individuals' personal data.
  The product page states nothing about personal data or terms
  (https://download.companieshouse.gov.uk/en_pscdata.html); the data-field description sits on
  a forum page (http://forum.aws.chdev.org/t/psc-streaming-api/529) which was not opened.
- Officers and PSC are available through the API (resources listed in the API reference).
- Position stated by Companies House: "You are responsible for complying with any applicable data
  protection, copyright and other legislation and regulations." The OGL and the Gazette licence
  both exclude personal data.
- **Recommendation for the register:** do not use the PSC snapshot or officer data at all. The
  cohort rule, features and outcomes can be built from company-level fields, filed accounts and
  company-level insolvency status, which keeps personal data out of the repository and out of the
  published cohort hash inputs. Gazette insolvency notices name individuals (directors,
  practitioners); store only company number, notice type and date.

## Recommended source

1. **Universe and static fields:** the monthly Free Company Data snapshot, archived on the day it
   is fetched, hashed, and committed by hash (the file itself is large and is third-party data).
2. **Features:** the Accounts Data Product (iXBRL), archived by us at fetch time.
3. **Outcomes:** Companies House `company_status`, `company_status_detail` and the insolvency
   resource via the API for firms that changed status or left the snapshot, cross-checked against
   Gazette corporate-insolvency notices (OGL v3.0, attribution required).
4. **Licence position to record:** Companies House register data, no stated conditions beyond
   third-party copyright, Crown insignia and data protection; Gazette under OGL v3.0.

## Not established

1. A formal licence for the Free Company Data, Accounts Data and PSC products. None is stated on
   their pages; whether the OGL applies to them is unconfirmed (the forum reply does not say).
2. Whether third-party copyright in filed accounts limits deriving and publishing numeric
   features or model scores. Needs the owner's decision and probably legal advice.
3. Whether the register can be described as a commercial product input without further
   permission. Companies House says it places no restriction; no written confirmation was sought.
4. The fields actually present in today's snapshot (the PDF is from 2012; no file was downloaded).
5. How many live companies have electronically filed accounts, and how many with iXBRL that
   parses into the balance sheet and profit and loss items the model needs. This sets the
   achievable cohort size that `docs/results/FINDINGS.md` §3 says is still unknown.
6. Whether Companies House retains older monthly basic-data snapshots online, and whether
   dissolved companies stay in a snapshot for any period after dissolution.
7. The descriptions of `date_of_cessation` and `date_of_creation` in the company profile, and the
   exact semantics of each insolvency date type.
8. Bulk-feed terms for Gazette data: the Gazette terms and conditions page says "Separate terms
   and conditions apply for using The Gazette data services." and those were not found. The fair
   use policy is now read (see Rate limits).
9. Companies House API terms of use: no separate terms page exists on the developer site (the
   guessed `/terms` and `/terms-and-conditions` URLs return 404; get-started, overview,
   authentication, developer-guidelines and the specs index state none). Its "Policies" footer
   link goes to https://www.gov.uk/guidance/search-the-companies-house-register, which holds an
   acceptable-use paragraph for the searchable databases, not API terms. The only API-specific
   conditions found are the rate limit and the right to ban applications that exceed it.
10. Delay between a legal event and its appearance on the register or in the Gazette, which sets
    how long after the 12-month horizon outcomes must be read.
11. Whether an API key and a "not supported" free product are acceptable as an operational
    dependency for a 12-month-plus programme (no service level is stated).

## Verification (2026-11-15, raw page text)

Method: each URL fetched with `curl -sL` (browser User-Agent), tags and scripts stripped with
Python's `html.parser`, whitespace and typographic quotes normalised, then each quoted sentence
searched as an exact substring. PDF read with `pdftotext`. "Verbatim" means found as an exact
string in the page text.

| URL | quotes checked | result |
| --- | --- | --- |
| https://www.gov.uk/government/publications/companies-house-accreditation-to-information-fair-traders-scheme/public-task-copyright-and-crown-copyright | "Material of this sort is subject to Crown copyright protection"; "is made available on GOV.UK under the Open Government Licence v3.0, except where otherwise stated."; section 2.2 "Companies House imposes no rules or requirements on how the information on the public register is used." and the section 47 / SI 1997/3032 sentence; section 3 "We place no restriction on how the information is used after purchase, other than the following:", the third-party copyright item and the Crown insignia item | all verbatim. **Differed:** the third item of section 3 was recorded in brackets as a paraphrase; the page reads "if information is used from guidance notes, the website, publications or statistical tables the customer must credit Companies House as the source of the information". Corrected. |
| https://forum.companieshouse.gov.uk/t/is-data-provided-under-the-ocs/4513 | "Companies House is not responsible for your use of the company data."; "You are responsible for complying with any applicable data protection, copyright and other legislation and regulations."; reply dated 4 Nov 2021 by the Service Owner for Get Company Information | verbatim. The "imposes no rules" sentence also appears in the reply, verbatim. The reply does not mention the OGL; the only OGL mention is in the question of 3 Nov 2021. |
| https://www.thegazette.co.uk/data | "All content is available under the Open Government Licence v3.0, except where otherwise stated."; "please note that this licence does not cover the re-use of personal data"; "Companies (from 1998): insolvency, ..."; formats PDF, TTL, RDF, JSON LD and a SPARQL endpoint; link to the Fair use policy | all verbatim. |
| https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ | grant sentence "The Licensor grants you a worldwide, royalty-free, perpetual, non-exclusive licence to use the Information subject to the conditions below."; attribution condition; "Contains public sector information licensed under the Open Government Licence v3.0."; exclusions (personal data, third party rights, logos and crests, patents, trade marks, design rights) | all verbatim except one. **Differed:** the commercial-use sentence has a comma after "other Information": "exploit the Information commercially and non-commercially for example, by combining it with other Information, or by including it in your own product or application." Corrected. |
| https://developer-specs.company-information.service.gov.uk/guides/rateLimiting | "You can make up to 600 requests within a five-minute period."; "429 Too Many Requests" | verbatim. |
| https://developer.company-information.service.gov.uk/developer-guidelines | 600 requests per "5 minute period"; ban sentence; higher-limit sentence | verbatim (new, added to Rate limits). |
| https://download.companieshouse.gov.uk/en_output.html | "a downloadable data snapshot"; "containing basic company data of live companies on the register"; "updated within 5 working days of the previous month end"; file list dated 2026-11-01, one 471Mb file or seven parts | verbatim. Terms sentence: "This snapshot is provided free of charge and will not be supported." No licence, copyright or OGL text on the page. |
| https://download.companieshouse.gov.uk/en_accountsdata.html | "a free downloadable ZIP file, which contains the individual data files"; "A new daily file will be added each morning over a 5-day period (Tuesday-Saturday)"; "will only be available to download for 60 calendar days"; "historic accounts data is available from the previous year's monthly files"; "Data is only available for electronically filed accounts" | all verbatim. Terms sentence: "Each data file is provided free of charge and is not supported." No licence text. |
| https://download.companieshouse.gov.uk/en_pscdata.html | "a downloadable data snapshot containing the full list of PSC's provided to Companies House"; "updated every morning before 10am GMT"; "All files are overwritten daily"; 33 parts dated 2026-11-15 | verbatim. Terms sentence: "This snapshot is provided free of charge and will not be supported." No licence text. |
| https://www.thegazette.co.uk/all-notices/notice | corporate notice types "Administration", "Creditors' voluntary liquidation", "Liquidation by the Court", "Members' voluntary liquidation", "Moratoria", "Receivership" | all present as exact strings. |
| https://resources.companieshouse.gov.uk/toolsToHelp/pdf/freeDataProductDataset.pdf | field names listed under "Fields in the basic-data snapshot"; PDF creation date 15 May 2012 | field names found (printed as "Dissolution Date" with a space); creation date confirmed. |
| https://www.thegazette.co.uk/fair-use-policy | newly read, see Rate limits | read in full. |
| https://www.thegazette.co.uk/terms-and-conditions | "Separate terms and conditions apply for using The Gazette data services." | verbatim; those data-services terms were not found. |
| https://developer.company-information.service.gov.uk/ (overview, get-started, authentication) and https://developer-specs.company-information.service.gov.uk/ | search for API terms of use | none stated; see Not established 9. |
| https://developer.company-information.service.gov.uk/terms and `/terms-and-conditions` | guessed URLs | HTTP 404. |
| https://www.gov.uk/guidance/search-the-companies-house-register (the developer site "Policies" link; redirects to `/guidance/searching-the-companies-house-register`) | acceptable use of searchable databases | read; "they are not intended to be a source for bulk downloads". No API terms. |

Not re-verified here: the field lists and enumerations taken from the OpenAPI resource pages
(company status values, insolvency case and date types). They are not licence or terms quotes
and the Verify for 6.1 does not depend on them; they were read from the specification files on
the first pass.

Outcome: every licence and terms quote that the Verify depends on is now confirmed verbatim on the
page itself, with two quotes corrected as above. The conclusions are unchanged: no stated licence
on the bulk product pages, no stated conditions on register use beyond third-party copyright and
the Crown insignia, Gazette under the OGL v3.0 without personal data.

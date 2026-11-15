# UK Companies House as the forward-register data source (task 6.1)

**Date checked: 2026-11-15.** Every claim below was read on the page named beside it on that
date; WebFetch returns a summary of the page, so the quoted phrases are the summariser's
extraction of the page text and each should be re-read on the page before anything external
rests on it (`docs/results/POSTMORTEM.md` records a summariser fabricating a table). The two
API resource pages and the data-fields PDF were fetched as files and read locally, so those
quotes are direct. Nothing was downloaded beyond product pages, one OpenAPI document and one
PDF; no account or API key was created, so nothing here comes from a live API call. Status of
every number: READ FROM A PAGE, not measured by us.

## Gaps first

- **No formal licence is stated on any bulk product page.** The free company data, accounts
  data and PSC pages carry no licence, copyright or terms section; their only terms-type
  sentence is that the product is free and "not supported". The Open Government Licence is
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
- Gazette: the data page points to a "Fair use policy" (https://www.thegazette.co.uk/fair-use-policy);
  it was not opened, so its limits are unknown.
- Arithmetic (ours, not a source figure): 600 requests per 5 minutes is 7,200 per hour. A status
  check of 100,000 firms at one request each takes about 14 hours of continuous calling, and
  fetching the insolvency resource as well doubles it. Use the bulk snapshot for the universe and
  the API only for the firms whose status changed or went missing.

## Licence

| source | stated terms | page |
| --- | --- | --- |
| Companies House publications | "Material of this sort is subject to Crown copyright protection" and "is made available on GOV.UK under the Open Government Licence v3.0, except where otherwise stated." | https://www.gov.uk/government/publications/companies-house-accreditation-to-information-fair-traders-scheme/public-task-copyright-and-crown-copyright |
| Public register information | "Companies House imposes no rules or requirements on how the information on the public register is used." Made available "by virtue of approvals issued by us in accordance with section 47 of the Copyright, Designs and Patents Act 1988 and Schedule 1 of the Database Regulations (SI 1997/3032)." | same page, section 2.2 |
| Re-use | "We place no restriction on how the information is used after purchase, other than the following: all customers, including bulk customers, must take their own legal advice regarding possible breach of third party copyright; customers cannot reproduce the Crown insignia; [credit Companies House if using guidance notes, website, publications or statistical tables]" | same page, section 3 |
| Developer forum, Service Owner for Get Company Information, 4 Nov 2021 | "Companies House imposes no rules or requirements on how the information on the public register is used." "Companies House is not responsible for your use of the company data." "You are responsible for complying with any applicable data protection, copyright and other legislation and regulations." The reply does not mention the OGL by name and does not confirm that it applies to API data. | https://forum.companieshouse.gov.uk/t/is-data-provided-under-the-ocs/4513 |
| The Gazette | "All content is available under the Open Government Licence v3.0, except where otherwise stated." "please note that this licence does not cover the re-use of personal data." | https://www.thegazette.co.uk/data |
| Open Government Licence v3.0 | "exploit the Information commercially and non-commercially for example, by combining it with other Information or by including it in your own product or application." Condition: "acknowledge the source of the Information in your product or application" using "Contains public sector information licensed under the Open Government Licence v3.0." Not covered: "personal data in the Information", "third party rights the Information Provider is not authorised to license", logos and crests, patents, trade marks and design rights. Grant: "a worldwide, royalty-free, perpetual, non-exclusive licence". | https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ |

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
8. The Gazette fair-use policy and any rate limit or bulk-feed terms for Gazette data.
9. Companies House API terms of use: the developer terms page returned 404 and the get-started
   page states none. The "Policies" link on the developer site was not followed.
10. Delay between a legal event and its appearance on the register or in the Gazette, which sets
    how long after the 12-month horizon outcomes must be read.
11. Whether an API key and a "not supported" free product are acceptable as an operational
    dependency for a 12-month-plus programme (no service level is stated).

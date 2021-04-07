---
title: Mechanistic Autoencoders for Patient-Specific Phosphoproteomic Models
keywords:
- markdown
- publishing
- manubot
lang: en-US
date-meta: '2021-04-06'
author-meta:
- Fabian Fröhlich
- Sara JC Gosline
- Jackson L. Chin
- Emek Demir
- Aaron S. Meyer
header-includes: |-
  <!--
  Manubot generated metadata rendered from header-includes-template.html.
  Suggest improvements at https://github.com/manubot/manubot/blob/master/manubot/process/header-includes-template.html
  -->
  <meta name="dc.format" content="text/html" />
  <meta name="dc.title" content="Mechanistic Autoencoders for Patient-Specific Phosphoproteomic Models" />
  <meta name="citation_title" content="Mechanistic Autoencoders for Patient-Specific Phosphoproteomic Models" />
  <meta property="og:title" content="Mechanistic Autoencoders for Patient-Specific Phosphoproteomic Models" />
  <meta property="twitter:title" content="Mechanistic Autoencoders for Patient-Specific Phosphoproteomic Models" />
  <meta name="dc.date" content="2021-04-06" />
  <meta name="citation_publication_date" content="2021-04-06" />
  <meta name="dc.language" content="en-US" />
  <meta name="citation_language" content="en-US" />
  <meta name="dc.relation.ispartof" content="Manubot" />
  <meta name="dc.publisher" content="Manubot" />
  <meta name="citation_journal_title" content="Manubot" />
  <meta name="citation_technical_report_institution" content="Manubot" />
  <meta name="citation_author" content="Fabian Fröhlich" />
  <meta name="citation_author_institution" content="Department of Systems Biology, Harvard Medical School" />
  <meta name="citation_author_orcid" content="0000-0002-5360-4292" />
  <meta name="twitter:creator" content="@fabfrohlich" />
  <meta name="citation_author" content="Sara JC Gosline" />
  <meta name="citation_author_institution" content="Pacific Northwest National Laboratories" />
  <meta name="citation_author_orcid" content="0000-0002-6534-4774" />
  <meta name="twitter:creator" content="@sargoshoe" />
  <meta name="citation_author" content="Jackson L. Chin" />
  <meta name="citation_author_institution" content="Department of Bioengineering, University of California, Los Angeles" />
  <meta name="citation_author" content="Emek Demir" />
  <meta name="citation_author_institution" content="Department of Molecular and Medical Genetics, Oregon Health &amp; Sciences Univerity" />
  <meta name="citation_author_orcid" content="0000-0002-3663-7113" />
  <meta name="citation_author" content="Aaron S. Meyer" />
  <meta name="citation_author_institution" content="Department of Bioengineering, University of California, Los Angeles" />
  <meta name="citation_author_institution" content="Department of Bioinformatics, University of California, Los Angeles" />
  <meta name="citation_author_institution" content="Jonsson Comprehensive Cancer Center, University of California, Los Angeles" />
  <meta name="citation_author_institution" content="Eli and Edythe Broad Center of Regenerative Medicine and Stem Cell Research, University of California, Los Angeles" />
  <meta name="citation_author_orcid" content="0000-0003-4513-1840" />
  <meta name="twitter:creator" content="@aarmey" />
  <link rel="canonical" href="https://meyer-lab.github.io/mechanismEncoder/" />
  <meta property="og:url" content="https://meyer-lab.github.io/mechanismEncoder/" />
  <meta property="twitter:url" content="https://meyer-lab.github.io/mechanismEncoder/" />
  <meta name="citation_fulltext_html_url" content="https://meyer-lab.github.io/mechanismEncoder/" />
  <meta name="citation_pdf_url" content="https://meyer-lab.github.io/mechanismEncoder/manuscript.pdf" />
  <link rel="alternate" type="application/pdf" href="https://meyer-lab.github.io/mechanismEncoder/manuscript.pdf" />
  <link rel="alternate" type="text/html" href="https://meyer-lab.github.io/mechanismEncoder/v/f851f7fa2f9e3b90c404c0b2352f9d9ff51e2931/" />
  <meta name="manubot_html_url_versioned" content="https://meyer-lab.github.io/mechanismEncoder/v/f851f7fa2f9e3b90c404c0b2352f9d9ff51e2931/" />
  <meta name="manubot_pdf_url_versioned" content="https://meyer-lab.github.io/mechanismEncoder/v/f851f7fa2f9e3b90c404c0b2352f9d9ff51e2931/manuscript.pdf" />
  <meta property="og:type" content="article" />
  <meta property="twitter:card" content="summary_large_image" />
  <link rel="icon" type="image/png" sizes="192x192" href="https://manubot.org/favicon-192x192.png" />
  <link rel="mask-icon" href="https://manubot.org/safari-pinned-tab.svg" color="#ad1457" />
  <meta name="theme-color" content="#ad1457" />
  <!-- end Manubot generated metadata -->
bibliography: []
manubot-output-bibliography: output/references.json
manubot-output-citekeys: output/citations.tsv
manubot-requests-cache-path: cache/requests-cache
manubot-clear-requests-cache: false
...




<small><em>
This manuscript
([permalink](https://meyer-lab.github.io/mechanismEncoder/v/f851f7fa2f9e3b90c404c0b2352f9d9ff51e2931/))
was automatically generated
from [meyer-lab/mechanismEncoder@f851f7f](https://github.com/meyer-lab/mechanismEncoder/tree/f851f7fa2f9e3b90c404c0b2352f9d9ff51e2931)
on April 6, 2021.
</em></small>

## Authors


+ **Fabian Fröhlich**<br>
    ORCID 
    [0000-0002-5360-4292](https://orcid.org/0000-0002-5360-4292)
    · Github
    [FFroehlich](https://github.com/FFroehlich)
    · twitter
    [fabfrohlich](https://twitter.com/fabfrohlich)<br>
  <small>
     Department of Systems Biology, Harvard Medical School
  </small>

+ **Sara JC Gosline**<br>
    ORCID 
    [0000-0002-6534-4774](https://orcid.org/0000-0002-6534-4774)
    · Github
    [sgosline](https://github.com/sgosline)
    · twitter
    [sargoshoe](https://twitter.com/sargoshoe)<br>
  <small>
     Pacific Northwest National Laboratories
  </small>

+ **Jackson L. Chin**<br>
    · Github
    [JacksonLChin](https://github.com/JacksonLChin)<br>
  <small>
     Department of Bioengineering, University of California, Los Angeles
  </small>

+ **Emek Demir**<br>
    ORCID 
    [0000-0002-3663-7113](https://orcid.org/0000-0002-3663-7113)
    · Github
    [emekdemir](https://github.com/emekdemir)<br>
  <small>
     Department of Molecular and Medical Genetics, Oregon Health & Sciences Univerity
  </small>

+ **Aaron S. Meyer**<br>
    ORCID 
    [0000-0003-4513-1840](https://orcid.org/0000-0003-4513-1840)
    · Github
    [aarmey](https://github.com/aarmey)
    · twitter
    [aarmey](https://twitter.com/aarmey)<br>
  <small>
     Department of Bioengineering, University of California, Los Angeles; Department of Bioinformatics, University of California, Los Angeles; Jonsson Comprehensive Cancer Center, University of California, Los Angeles; Eli and Edythe Broad Center of Regenerative Medicine and Stem Cell Research, University of California, Los Angeles
  </small>



## Abstract {.page_break_before}

Proteomic data provides measurements that are uniquely close to the mechanism of action for many cancer therapies. As such, it can provide an unmatched perspective into the mechanism of drug action and resistance. At the same time, extracting the source of patient-to-patient differences in proteomic measurements and understanding its relevance for drug sensitivity is extremely challenging. Correlative analyses are most common but are difficult to mechanistically interpret.


## Introduction

Proteomic data provides measurements that are uniquely close to the mechanism of action for many cancer therapies. As such, it can provide an unmatched perspective into the mechanism of drug action and resistance [@DOI:10.1016/j.xcrm.2020.100004; @DOI:10.1016/j.cell.2019.10.007]. At the same time, extracting the source of patient-to-patient differences in proteomic measurements and understanding its relevance for drug sensitivity is extremely challenging. Correlative analyses are most common but are difficult to mechanistically interpret.

Mechanistic models are uniquely powerful for identifying the drivers of differences within measurements, integrating our prior knowledge, and interpreting data. However, a key question that limits their use for patient data is how to handle patient-to-patient differences. Constructing multiple patient-specific models is infeasible due to the limited data for each patient. Alternatively, universal models that use patient invariant and patient-specific parameters to integrate data across multiple individuals have been proposed [@DOI:10.1016/j.cels.2018.10.013]. However, how to estimate these patient-specific parameters is challenging as genetic and microenvironmental context influences signaling pathways in complex, non-linear, and often poorly understood ways.

At its core, the challenge of integrating mechanistic models with patient-derived measurements is an issue of how to account for patient-to-patient variation. Mechanistic dynamical models have been widely applied to data of all types but are used where the sources of variation among measurements can be explicitly identified and modeled. By contrast, variation among individuals can arise through both factors that can easily be identified, like changes in the abundance of the species being modeling, and endless other molecular and physiological factors that cannot be usefully enumerated in a mechanistic approach. Still, the structure of mechanistic models provides important constraints on the behavior of molecular pathways and interpretability that is missing from purely data-driven statistical methods.

To address this issue, we propose a model structure that is based on a variational autoencoder. Autoencoders are neural networks that embed data into low dimensional latent feature space by feeding the data through encoding and decoding layers [@DOI:10.1126/science.1127647]. The extracted latent features then provide a reduced representation of patient-patient similarity. We integrate mechanistic information by partly replacing the decoder layers in the network with a coarse-grained mechanistic model, where the encoded, latent representation of the data defines the patient-specific parameters of the universal ordinary differential equation (ODE) model. We apply this to AML patient samples, where proteomic and phosphoproteomic measurements with high tumor purity can be collected. This model structure enables mechanistic interpretation of these data; more robust latent space representations of patient relationships; and integration of prior knowledge, other data sources such as *in vitro* experiments or other data types, and clinical measurements. Mechanistic autoencoders, therefore, offer a general solution to building mechanistic models in the presence of unexplained sample variation, such as from clinical samples.


## Results

### MAEs mechanistically account for patient-to-patient variation

![**XXX.** A) XXX. B) XXX.](images/schematic.svg "Figure Cartoon"){#fig:cartoon width="100%"}

- Schematic of autoencoder structure
- Cartoon description of other encoder structures
- Part of this would be to speak to the generality of the approach

### Figure 2

![**XXX.** A) XXX. B) XXX.](images/schematic.svg "Figure Cartoon"){#fig:proteomic width="100%"}

Initial plot of proteomic data (clustergram?) - see #23
Data-driven selection of network nodes from OHSU

### Figure 3

![**XXX.** A) XXX. B) XXX.](images/schematic.svg "Figure Cartoon"){#fig:model width="100%"}

Training against actual data
Description of fit model

### Figure 4

![**XXX.** A) XXX. B) XXX.](images/schematic.svg "Figure Cartoon"){#fig:cellLine width="100%"}

Cell line perturbation
Description of that data

### Figure 5

![**XXX.** A) XXX. B) XXX.](images/schematic.svg "Figure Cartoon"){#fig:validation width="100%"}

Model/validation comparison


## Discussion

- Emphasize generality of approach
- Future possibilities
    - True validation in patient-derived samples
    - Pan-cancer modeling
    - Infer structure ala perturbation biology / neural ODEs?
- Cover other forms of mechanistic / data-driven integration?




FROM PROPOSAL

This project has the potential to enable routine use of mechanistic models to analyze clinical proteomics measurements. As such, one can easily envision applying a similar technique across many different cancer types as well as other diseases. It is hard to overstate the potential impact, as this can convert these measurements to (1) exacting predictions of which components to target in individual patients and (2) provide a mechanism-grounded view of patient-to-patient variation.

## Methods

### Data collection

Sara should likely fill this in.

### Basic autoencoder implementation

This would be Jackson's work.

### Mechanistic model implementation and integration

Fabian knows this.

### Pathway Commons analysis




## Acknowledgements

This work was supported by an administrative supplement to NIH U01-CA215709 to A.S.M. The authors declare no competing financial interests.

## Author contributions statement

XXX. J.L.C. implemented and analyzed the standard autoencoder as a baseline. All authors wrote the paper.


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>

---
title: "Predicting Marijuana Use Disorder: A Machine Learning Approach"
author: "Jiashu Liu"
date: "2024-06-16"
output:
  html_document:
    keep_md: true
    df_print: paged
    code_download: true
    theme: journal
    code_folding: hide
    css: styles.css
font-family: Times New Roman
---




```r
# Load libraries
library(tidyverse)
library(dplyr)
library(readr)
library(ggplot2)
library(httr)
library(jsonlite)
library(foreach)
library(psych)
library(patchwork)
library(gt) # for better table output
library(gridExtra)
library(ROCR)
library(ranger)
library(randomForest)
#library(pak)
#pak::pak("caret")
library(caret)
library(e1071)
library(nnet)
library(dummy)
library(gbm)
```

# Introduction 

Based on the previous group project, we changed some methods. 

# Data

The data utilized in this project comes from the 2022 National Survey on Drug Use and Health (NSDUH) Releases. This survey provides nationally representative data on tobacco, alcohol, and drug use; substance use disorders; mental health issues; and the receipt of substance use and mental health treatment among the civilian, non-institutionalized population aged 12 and older in the United States. Compared to the 2021 NSDUH data, the 2022 version updated questions related to substance use treatment and mental health treatment, vaping of nicotine and marijuana, different methods of marijuana use, and the use of illegally made fentanyl (IMF). Most importantly, the 2022 NSDUH data is less affected by the COVID-19 pandemic than the 2021 data, which was a significant consideration in our previous project. In-person data collection for the 2022 NSDUH was only restricted in January 2022, resulting in a higher proportion of in-person interviews in 2022 compared to 2021.


```r
# Load data -- NSDUH 2022
load("/Users/jiashuliu/Desktop/Projects/substance_use_disorder/data/NSDUH_2022.RData")
sud_2022 <- read_csv("/Users/jiashuliu/Desktop/Projects/substance_use_disorder/data/sud_2022.csv")
```

### Outcome Variable 

The outcome variable SUD_MJ is based on the DSM-5 Diagnostic Criteria for diagnosing and classifying substance use disorders. I reviewed questions from the marijuana use disorder section of the 2022 NSDUH data and identified the most relevant questions corresponding to the DSM-5 criteria. An SUD_MJ value of 1 indicates that the respondent has some level of marijuana use disorder, while a value of 0 indicates the absence of such a disorder. 


```r
# If any of the "udmj" variables have a value of 1, we set SUD_MJ as 1, otherwise SUD_MJ is 0.
NSDUH_2022_full <- NSDUH_2022 %>%
  mutate(across(starts_with("udmj"), 
                ~if_else(. %in% c(1, 2), 
                         if_else(. == 1, 1, 0), 
                         NA))) %>% 
  select(-"udmjavwothr") %>% 
  mutate(SUD_MJ = rowSums(select(., starts_with("udmj")), na.rm = TRUE)>=1, 
         SUD_MJ = if_else(SUD_MJ, 1, 0))
```

### Predictors

There are 14 selected predictors, including age, sex, race, whether the respondent has health problems, marital status, highest degree obtained, school attendance, employment status, number of people in the household, number of children under 18 in the household, number of elderly people over 65 in the household, health insurance status, family income level, and mental health status.

All the socio-demographic variables in the survey are provided as categorical variables with various levels. I followed the methodology used in the original survey data but reorganized some variables into more general levels to make the data easier to analyze and interpret. For example, the predictor age (AGE3 in the survey data) originally had 10 levels. I consolidated these into four levels: 1 represents adolescents under age 18, 2 represents young adults aged 18 to 29, 3 indicates middle-aged individuals aged 30 to 64, and 4 represents elderly individuals aged 65 and older. The final cleaned dataset 


```r
codebook = data.frame(Variable = c("Age", "Sex", "Race", "Health", "Marital", "Degree", "Now going to school or not?", "Employment", "Persons in Household", "Kids age<18 in Household", "Elderly age>65 in Household", "Health Insurance", "Income: family income", "Mentalhealth: combined score of K6 questions"), 
                      Meaning = c("1=Adolescent: 18-, 2=Young Adult: 18-29, 3=Middle Age: 30-64, 4=Elderly: 65+", "0=Female, 1=Male", "1=NonHisp White, 2=NonHisp Black/Afr Am, 3=NonHisp Native Am/AK Native, 4=NonHisp Native HI/Other Pac Isl, 5=NonHisp Asian, 6=NonHisp more than one race, 7=Hispanic", "0=w/o health problem: excellent/very good/good, 1=with health problem: fair/poor", "0=never been married/cannot married<=14, 1=married, 2=widowed/divorced/separated", "1=w/o high school, 2=high school degree, 3=associate's degree/college graduate or higher", "1/11 = now going to school,  0=No, other is NA", "1=employed full time, 2=employed part time, 3=unemployed, 4=Other(incl. not in labor force)", "range 1-5, 6=6 or more people in household", "0=No children under 18 , 1=One child under 18, 2=Two children under 18, 3=Three or more children under 18.", "0=No people 65 or older in household, 1 = One person 65 or older in household, 2 = Two or more people 65 or older in household", "0=w/o health insurance, 1=health insurance", "1=poverty:20000-, 2=middle:74999-, 3=wealth:75000+", "range = 0 - 24, na:Aged 12-17"))
codebook %>%
  gt() %>%
  tab_header(
    title = md("**Codebook**")
  ) %>%
  tab_style(
    style = cell_fill(color = "aliceblue"),
    locations = cells_body(
      rows = Variable %in% c("Age", "Race", "Marital", "Now going to school or not?", "Persons in Household", "Elderly age>65 in Household", "Income: family income"))
  ) %>%
  tab_style(
    style = cell_fill(color = "skyblue"),
    locations = cells_body(
      rows = !(Variable %in% c("Age", "Race", "Marital", "Now going to school or not?", "Persons in Household", "Elderly age>65 in Household", "Income: family income"))
    )
  ) %>%
  tab_options(
    table.font.size = px(13L)
  )
```

```{=html}
<div id="bxlxuckeks" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>#bxlxuckeks table {
  font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#bxlxuckeks thead, #bxlxuckeks tbody, #bxlxuckeks tfoot, #bxlxuckeks tr, #bxlxuckeks td, #bxlxuckeks th {
  border-style: none;
}

#bxlxuckeks p {
  margin: 0;
  padding: 0;
}

#bxlxuckeks .gt_table {
  display: table;
  border-collapse: collapse;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 13px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#bxlxuckeks .gt_caption {
  padding-top: 4px;
  padding-bottom: 4px;
}

#bxlxuckeks .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#bxlxuckeks .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#bxlxuckeks .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#bxlxuckeks .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#bxlxuckeks .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#bxlxuckeks .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#bxlxuckeks .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#bxlxuckeks .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#bxlxuckeks .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#bxlxuckeks .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#bxlxuckeks .gt_spanner_row {
  border-bottom-style: hidden;
}

#bxlxuckeks .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  text-align: left;
}

#bxlxuckeks .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#bxlxuckeks .gt_from_md > :first-child {
  margin-top: 0;
}

#bxlxuckeks .gt_from_md > :last-child {
  margin-bottom: 0;
}

#bxlxuckeks .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#bxlxuckeks .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}

#bxlxuckeks .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}

#bxlxuckeks .gt_row_group_first td {
  border-top-width: 2px;
}

#bxlxuckeks .gt_row_group_first th {
  border-top-width: 2px;
}

#bxlxuckeks .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#bxlxuckeks .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}

#bxlxuckeks .gt_first_summary_row.thick {
  border-top-width: 2px;
}

#bxlxuckeks .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#bxlxuckeks .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#bxlxuckeks .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#bxlxuckeks .gt_last_grand_summary_row_top {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: double;
  border-bottom-width: 6px;
  border-bottom-color: #D3D3D3;
}

#bxlxuckeks .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#bxlxuckeks .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#bxlxuckeks .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#bxlxuckeks .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#bxlxuckeks .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#bxlxuckeks .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#bxlxuckeks .gt_left {
  text-align: left;
}

#bxlxuckeks .gt_center {
  text-align: center;
}

#bxlxuckeks .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#bxlxuckeks .gt_font_normal {
  font-weight: normal;
}

#bxlxuckeks .gt_font_bold {
  font-weight: bold;
}

#bxlxuckeks .gt_font_italic {
  font-style: italic;
}

#bxlxuckeks .gt_super {
  font-size: 65%;
}

#bxlxuckeks .gt_footnote_marks {
  font-size: 75%;
  vertical-align: 0.4em;
  position: initial;
}

#bxlxuckeks .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}

#bxlxuckeks .gt_indent_1 {
  text-indent: 5px;
}

#bxlxuckeks .gt_indent_2 {
  text-indent: 10px;
}

#bxlxuckeks .gt_indent_3 {
  text-indent: 15px;
}

#bxlxuckeks .gt_indent_4 {
  text-indent: 20px;
}

#bxlxuckeks .gt_indent_5 {
  text-indent: 25px;
}
</style>
<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
  <thead>
    <tr class="gt_heading">
      <td colspan="2" class="gt_heading gt_title gt_font_normal gt_bottom_border" style><strong>Codebook</strong></td>
    </tr>
    
    <tr class="gt_col_headings">
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="Variable">Variable</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="Meaning">Meaning</th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr><td headers="Variable" class="gt_row gt_left" style="background-color: #F0F8FF;">Age</td>
<td headers="Meaning" class="gt_row gt_left" style="background-color: #F0F8FF;">1=Adolescent: 18-, 2=Young Adult: 18-29, 3=Middle Age: 30-64, 4=Elderly: 65+</td></tr>
    <tr><td headers="Variable" class="gt_row gt_left" style="background-color: #87CEEB;">Sex</td>
<td headers="Meaning" class="gt_row gt_left" style="background-color: #87CEEB;">0=Female, 1=Male</td></tr>
    <tr><td headers="Variable" class="gt_row gt_left" style="background-color: #F0F8FF;">Race</td>
<td headers="Meaning" class="gt_row gt_left" style="background-color: #F0F8FF;">1=NonHisp White, 2=NonHisp Black/Afr Am, 3=NonHisp Native Am/AK Native, 4=NonHisp Native HI/Other Pac Isl, 5=NonHisp Asian, 6=NonHisp more than one race, 7=Hispanic</td></tr>
    <tr><td headers="Variable" class="gt_row gt_left" style="background-color: #87CEEB;">Health</td>
<td headers="Meaning" class="gt_row gt_left" style="background-color: #87CEEB;">0=w/o health problem: excellent/very good/good, 1=with health problem: fair/poor</td></tr>
    <tr><td headers="Variable" class="gt_row gt_left" style="background-color: #F0F8FF;">Marital</td>
<td headers="Meaning" class="gt_row gt_left" style="background-color: #F0F8FF;">0=never been married/cannot married&lt;=14, 1=married, 2=widowed/divorced/separated</td></tr>
    <tr><td headers="Variable" class="gt_row gt_left" style="background-color: #87CEEB;">Degree</td>
<td headers="Meaning" class="gt_row gt_left" style="background-color: #87CEEB;">1=w/o high school, 2=high school degree, 3=associate's degree/college graduate or higher</td></tr>
    <tr><td headers="Variable" class="gt_row gt_left" style="background-color: #F0F8FF;">Now going to school or not?</td>
<td headers="Meaning" class="gt_row gt_left" style="background-color: #F0F8FF;">1/11 = now going to school,  0=No, other is NA</td></tr>
    <tr><td headers="Variable" class="gt_row gt_left" style="background-color: #87CEEB;">Employment</td>
<td headers="Meaning" class="gt_row gt_left" style="background-color: #87CEEB;">1=employed full time, 2=employed part time, 3=unemployed, 4=Other(incl. not in labor force)</td></tr>
    <tr><td headers="Variable" class="gt_row gt_left" style="background-color: #F0F8FF;">Persons in Household</td>
<td headers="Meaning" class="gt_row gt_left" style="background-color: #F0F8FF;">range 1-5, 6=6 or more people in household</td></tr>
    <tr><td headers="Variable" class="gt_row gt_left" style="background-color: #87CEEB;">Kids age&lt;18 in Household</td>
<td headers="Meaning" class="gt_row gt_left" style="background-color: #87CEEB;">0=No children under 18 , 1=One child under 18, 2=Two children under 18, 3=Three or more children under 18.</td></tr>
    <tr><td headers="Variable" class="gt_row gt_left" style="background-color: #F0F8FF;">Elderly age&gt;65 in Household</td>
<td headers="Meaning" class="gt_row gt_left" style="background-color: #F0F8FF;">0=No people 65 or older in household, 1 = One person 65 or older in household, 2 = Two or more people 65 or older in household</td></tr>
    <tr><td headers="Variable" class="gt_row gt_left" style="background-color: #87CEEB;">Health Insurance</td>
<td headers="Meaning" class="gt_row gt_left" style="background-color: #87CEEB;">0=w/o health insurance, 1=health insurance</td></tr>
    <tr><td headers="Variable" class="gt_row gt_left" style="background-color: #F0F8FF;">Income: family income</td>
<td headers="Meaning" class="gt_row gt_left" style="background-color: #F0F8FF;">1=poverty:20000-, 2=middle:74999-, 3=wealth:75000+</td></tr>
    <tr><td headers="Variable" class="gt_row gt_left" style="background-color: #87CEEB;">Mentalhealth: combined score of K6 questions</td>
<td headers="Meaning" class="gt_row gt_left" style="background-color: #87CEEB;">range = 0 - 24, na:Aged 12-17</td></tr>
  </tbody>
  
  
</table>
</div>
```


```r
# Predictors -- Demographics
# 1) age (1=Adolescent: 18-, 2=Young Adult: 18-29, 3=Middle Age: 30-64, 4=Elderly: 65+)
NSDUH_2022_full <- NSDUH_2022_full %>% 
  mutate(age = case_when(AGE3 %in% c(1:3) ~ 1,
                         AGE3 %in% c(4:8) ~ 2,
                         AGE3 %in% c(9:10) ~ 3,
                         TRUE ~ 4))
# 2) sex (0=Female, 1=Male)
NSDUH_2022_full <- NSDUH_2022_full %>% 
  mutate(sex = if_else(irsex == 2,0,1))

# 3) race (1=NonHisp White, 2=NonHisp Black/Afr Am, 3=NonHisp Native Am/AK Native, 4=NonHisp Native HI/Other Pac Isl, 5=NonHisp Asian, 6=NonHisp more than one race, 7=Hispanic)
NSDUH_2022_full <- NSDUH_2022_full %>% 
  mutate(race = NEWRACE2)
  #mutate(race = case_when(NEWRACE2 %in% c(2:6) ~ 2,
                         # NEWRACE2 == 7 ~ 3,
                          #TRUE ~ 1))

# 4) health (0=w/o health problem: excellent/very good/good, 1=with health problem: fair/poor)
NSDUH_2022_full <- NSDUH_2022_full %>% 
  mutate(health = case_when(health %in% c(1:3) ~ 0,
                            health %in% c(4:5) ~ 1,
                            TRUE ~ NA))

# 5) marital (0=never been married/cannot married<=14, 1=married, 2=widowed/divorced/separated)
NSDUH_2022_full <- NSDUH_2022_full %>% 
  mutate(marital = case_when(irmarit %in% c(4,99) ~ 0,
                             irmarit %in% c(2:3) ~ 2,
                             TRUE ~ 1))
```


```r
# Predictors -- Education
# 6) degree (1=w/o high school, 2=high school degree, 3=associate's degree/college graduate or higher)
NSDUH_2022_full <- NSDUH_2022_full %>% 
  mutate(degree = case_when(IREDUHIGHST2 %in% c(1:7) ~ 1,
                            IREDUHIGHST2 %in% c(8:9) ~ 2,
                            TRUE ~ 3))

# 7) Now going to school or not? (1/11 = now going to school,  0=No, other is NA)
NSDUH_2022_full <- NSDUH_2022_full %>% 
  mutate(student = case_when(eduschlgo %in% c(1, 11) ~ 1,
                             eduschlgo == 2 ~ 0,
                             TRUE ~ NA))
```


```r
# Predictors: Employment and Houshold Composition
# 8) employ (1=employed full time, 2=employed part time, 3=unemployed, 4=Other(incl. not in labor force))
NSDUH_2022_full <- NSDUH_2022_full %>% 
  mutate(employ = case_when(WRKSTATWK2 %in% c(1,6) ~ 1,
                            WRKSTATWK2 %in% c(2:3) ~ 2,
                            WRKSTATWK2 %in% c(4,9) ~ 3,
                            TRUE ~ 4))

# 9) persons in household (range 1-5, 6=6 or more people in household)
NSDUH_2022_full <- NSDUH_2022_full %>% mutate(family = IRHHSIZ2)

# 10) kids age<18 in Household
# 0 = No children under 18 
# 1 = One child under 18
# 2 = Two children under 18 
# 3 = Three or more children under 18.
NSDUH_2022_full <- NSDUH_2022_full %>% mutate(kid = IRKI17_2 - 1)

# 11) elderly age>65 in Household (range 0-1, 2=2 or more elders in household)
# 0 = No people 65 or older in household
# 1 = One person 65 or older in household
# 2 = Two or more people 65 or older in household
NSDUH_2022_full <- NSDUH_2022_full %>% mutate(elderly = IRHH65_2-1)
```


```r
# Predictors: Health and Income 
# 12) health_insur (0=w/o health insurance, 1=health insurance)
NSDUH_2022_full <- NSDUH_2022_full %>% 
  mutate(health_insur = case_when(
    irmedicr == 1 | irmcdchp == 1 | irchmpus == 1 | irprvhlt == 1 ~ 1,
    irothhlt == 1 ~ 1,
    irothhlt == 2 ~ 0,
    irothhlt == 99 ~ NA,
    TRUE ~ 0
  ))

# 13) income: family income (1=poverty:20000-, 2=middle:74999-, 3=wealth:75000+)
NSDUH_2022_full <- NSDUH_2022_full %>% 
  mutate(income = case_when(IRFAMIN3 %in% c(1:2) ~ 1,
                            IRFAMIN3 %in% c(3:6) ~ 2,
                            TRUE ~ 3))

# 14) mentalhealth: combined score of K6 questions (range = 0 - 24, na:Aged 12-17) 
NSDUH_2022_full <- NSDUH_2022_full %>% 
  mutate(k1 = case_when(IRDSTCHR30 == 1 ~ 4,
                         IRDSTCHR30 == 2 ~ 3,
                         IRDSTCHR30 == 3 ~ 2,
                         IRDSTCHR30 == 4 ~ 1,
                         IRDSTCHR30 == 5 ~ 0,
                         TRUE ~ 99),
         k2 = case_when(IRDSTEFF30 == 1 ~ 4,
                         IRDSTEFF30 == 2 ~ 3,
                         IRDSTEFF30 == 3 ~ 2,
                         IRDSTEFF30 == 4 ~ 1,
                         IRDSTEFF30 == 5 ~ 0,
                         TRUE ~ 99),
         k3 = case_when(IRDSTHOP30 == 1 ~ 4,
                         IRDSTHOP30 == 2 ~ 3,
                         IRDSTHOP30 == 3 ~ 2,
                         IRDSTHOP30 == 4 ~ 1,
                         IRDSTHOP30 == 5 ~ 0,
                         TRUE ~ 99),
         k4 = case_when(IRDSTNGD30 == 1 ~ 4,
                         IRDSTNGD30 == 2 ~ 3,
                         IRDSTNGD30 == 3 ~ 2,
                         IRDSTNGD30 == 4 ~ 1,
                         IRDSTNGD30 == 5 ~ 0,
                         TRUE ~ 99),
         k5 = case_when(IRDSTNRV30 == 1 ~ 4,
                         IRDSTNRV30 == 2 ~ 3,
                         IRDSTNRV30 == 3 ~ 2,
                         IRDSTNRV30 == 4 ~ 1,
                         IRDSTNRV30 == 5 ~ 0,
                         TRUE ~ 99),
         k6 = case_when(IRDSTRST30 == 1 ~ 4,
                         IRDSTRST30 == 2 ~ 3,
                         IRDSTRST30 == 3 ~ 2,
                         IRDSTRST30 == 4 ~ 1,
                         IRDSTRST30 == 5 ~ 0,
                         TRUE ~ 99),
         mentalhealth = case_when(k1 == 99 | k2 == 99 | k3 == 99 | k4 == 99 | 
                                    k5 == 99 | k6 == 99 ~  NA,
                                  TRUE ~ k1+k2+k3+k4+k5+k6))
# For each of the six items listed above, responses of "all of the time" were coded 4, 
#"most of the time" were coded 3, "some of the time" were coded 2, "a little of the time" 
#were coded 1, and "none of the time" were coded 0. These assigned values were summed 
#across the six items to calculate a total score for mentalhealth.
```


```r
# Create new dataset and drop all the NA values
data_cleaned <- NSDUH_2022_full %>%
  select(age, sex, race, health, marital, degree, 
         student, employ, family, kid, elderly, health_insur, 
         income, mentalhealth, SUD_MJ) %>%
  drop_na()
# check NAs
# anyNA(data_cleaned)
# New csv file
# write_csv(data_cleaned,"/Users/jiashuliu/Desktop/Projects/substance_use_disorder/data/sud_2022.csv")
```

# Exploratory Data Analysis

### Data Visualization

**Age**
The bar plot of the age variable indicates that young adults aged 18 to 29 have the highest prevalence of marijuana use disorder.

```r
# age (1=Adolescent: 18-, 2=Young Adult: 18-29, 3=Middle Age: 30-64, 4=Elderly: 65+)
table(sud_2022$age)
```

```
## 
##     2     3     4 
## 23060 17273  4998
```

```r
p1.1 <- sud_2022 %>% 
  ggplot(aes(x = factor(age, levels = 1:4, labels = c("Adolescent", "Young Adult", "Middle Age","Elderly")), fill = factor(SUD_MJ))) +
  geom_bar(alpha = 0.5, position = "dodge") +
  scale_fill_manual(values = c("#619CFF", "#FF595E"),name = "SUD_MJ") +
  labs(x = "age", y = "Count") +
  theme_minimal()
p1.1
```

![](Final_Writeup_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

**Sex**
From the bar plot, we can see that the percentage of males having a marijuana use disorder in 2022 is slightly higher than that of females, with 17% of males and 12.5% of females affected.

```r
# Calculate the percentage of SUD_MJ for each sex and each level of SUD_MJ
percentage <- sud_2022 %>%
  group_by(sex, SUD_MJ) %>%
  summarize(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)

percentage <- percentage %>%
  mutate(sex = factor(sex, levels = c(0, 1), labels = c("Female", "Male")),
         SUD_MJ = factor(SUD_MJ, levels = c(0, 1), labels = c("0", "1")))

# Plot SUD_MJ percentages in each gender group
p2 <- sud_2022 %>%
  ggplot(aes(x = factor(sex, levels = c(0, 1), labels = c("Female", "Male")), fill = factor(SUD_MJ))) +
  geom_bar(alpha = 0.5, position = "dodge") +
  scale_fill_manual(values = c("#619CFF", "#FF595E"), name = "SUD_MJ") +
  labs(x = "Sex", y = "Count") +
  theme_minimal() +
  geom_text(data = percentage, aes(x = sex, y = count, label = paste0(round(percentage, 1), "%")),
            position = position_dodge(width = 0.9), vjust = -0.5)

p2
```

![](Final_Writeup_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

**Race**
The following table summarizes the distribution of marijuana use disorder (SUD_MJ) among different racial groups. The count and percentage are provided for each combination of race and SUD_MJ status. The "NonHisp Asian" group has the lowest percentage (5.2%) of individuals with SUD_MJ, while the "NonHisp Native Am/AK Native" group has the highest percentage (27.3%). The "NonHisp White" group has the second-lowest percentage (13.7%) of individuals with SUD_MJ, and the "NonHisp More Than One Race" group has the second-highest percentage (23.1%). The "NonHisp Black/Afr Am" group has a percentage of 17.9% of individuals with SUD_MJ, which is higher than the "Hispanic" group, which has 15.1% of individuals with SUD_MJ.

```r
race_dat <- sud_2022 %>%
  mutate(race = factor(race, levels = 1:7, labels = c("NonHisp White", "NonHisp Black/Afr Am", "NonHisp Native Am/AK Native", "NonHisp Native HI/Other Pac Isl", "NonHisp Asian", "NonHisp more than one race", "Hispanic")))
race_summary <- race_dat %>%
  group_by(race, SUD_MJ) %>%
  summarize(count = n(), .groups = 'drop') %>%
  group_by(race) %>%
  mutate(percentage = round(count / sum(count) * 100, 1)) %>%
  ungroup()
highlighted_table <- race_summary %>%
  gt() %>%
  tab_style(
    style = cell_fill(color = "lightyellow"),
    locations = cells_body(
      rows = race == "NonHisp Native Am/AK Native" & SUD_MJ == 1
    )
  ) %>% 
  tab_style(
    style = cell_fill(color = "honeydew"),
    locations = cells_body(
      rows = race == "NonHisp Asian" & SUD_MJ == 1
    )
  )
highlighted_table
```

```{=html}
<div id="jzgtqruehs" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>#jzgtqruehs table {
  font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#jzgtqruehs thead, #jzgtqruehs tbody, #jzgtqruehs tfoot, #jzgtqruehs tr, #jzgtqruehs td, #jzgtqruehs th {
  border-style: none;
}

#jzgtqruehs p {
  margin: 0;
  padding: 0;
}

#jzgtqruehs .gt_table {
  display: table;
  border-collapse: collapse;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#jzgtqruehs .gt_caption {
  padding-top: 4px;
  padding-bottom: 4px;
}

#jzgtqruehs .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#jzgtqruehs .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#jzgtqruehs .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#jzgtqruehs .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#jzgtqruehs .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#jzgtqruehs .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#jzgtqruehs .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#jzgtqruehs .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#jzgtqruehs .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#jzgtqruehs .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#jzgtqruehs .gt_spanner_row {
  border-bottom-style: hidden;
}

#jzgtqruehs .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  text-align: left;
}

#jzgtqruehs .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#jzgtqruehs .gt_from_md > :first-child {
  margin-top: 0;
}

#jzgtqruehs .gt_from_md > :last-child {
  margin-bottom: 0;
}

#jzgtqruehs .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#jzgtqruehs .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}

#jzgtqruehs .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}

#jzgtqruehs .gt_row_group_first td {
  border-top-width: 2px;
}

#jzgtqruehs .gt_row_group_first th {
  border-top-width: 2px;
}

#jzgtqruehs .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#jzgtqruehs .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}

#jzgtqruehs .gt_first_summary_row.thick {
  border-top-width: 2px;
}

#jzgtqruehs .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#jzgtqruehs .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#jzgtqruehs .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#jzgtqruehs .gt_last_grand_summary_row_top {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: double;
  border-bottom-width: 6px;
  border-bottom-color: #D3D3D3;
}

#jzgtqruehs .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#jzgtqruehs .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#jzgtqruehs .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#jzgtqruehs .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#jzgtqruehs .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#jzgtqruehs .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#jzgtqruehs .gt_left {
  text-align: left;
}

#jzgtqruehs .gt_center {
  text-align: center;
}

#jzgtqruehs .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#jzgtqruehs .gt_font_normal {
  font-weight: normal;
}

#jzgtqruehs .gt_font_bold {
  font-weight: bold;
}

#jzgtqruehs .gt_font_italic {
  font-style: italic;
}

#jzgtqruehs .gt_super {
  font-size: 65%;
}

#jzgtqruehs .gt_footnote_marks {
  font-size: 75%;
  vertical-align: 0.4em;
  position: initial;
}

#jzgtqruehs .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}

#jzgtqruehs .gt_indent_1 {
  text-indent: 5px;
}

#jzgtqruehs .gt_indent_2 {
  text-indent: 10px;
}

#jzgtqruehs .gt_indent_3 {
  text-indent: 15px;
}

#jzgtqruehs .gt_indent_4 {
  text-indent: 20px;
}

#jzgtqruehs .gt_indent_5 {
  text-indent: 25px;
}
</style>
<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
  <thead>
    <tr class="gt_col_headings">
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="race">race</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="SUD_MJ">SUD_MJ</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="count">count</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="percentage">percentage</th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr><td headers="race" class="gt_row gt_center">NonHisp White</td>
<td headers="SUD_MJ" class="gt_row gt_right">0</td>
<td headers="count" class="gt_row gt_right">23718</td>
<td headers="percentage" class="gt_row gt_right">86.3</td></tr>
    <tr><td headers="race" class="gt_row gt_center">NonHisp White</td>
<td headers="SUD_MJ" class="gt_row gt_right">1</td>
<td headers="count" class="gt_row gt_right">3772</td>
<td headers="percentage" class="gt_row gt_right">13.7</td></tr>
    <tr><td headers="race" class="gt_row gt_center">NonHisp Black/Afr Am</td>
<td headers="SUD_MJ" class="gt_row gt_right">0</td>
<td headers="count" class="gt_row gt_right">4252</td>
<td headers="percentage" class="gt_row gt_right">82.1</td></tr>
    <tr><td headers="race" class="gt_row gt_center">NonHisp Black/Afr Am</td>
<td headers="SUD_MJ" class="gt_row gt_right">1</td>
<td headers="count" class="gt_row gt_right">927</td>
<td headers="percentage" class="gt_row gt_right">17.9</td></tr>
    <tr><td headers="race" class="gt_row gt_center">NonHisp Native Am/AK Native</td>
<td headers="SUD_MJ" class="gt_row gt_right">0</td>
<td headers="count" class="gt_row gt_right">437</td>
<td headers="percentage" class="gt_row gt_right">72.7</td></tr>
    <tr><td headers="race" class="gt_row gt_center" style="background-color: #FFFFE0;">NonHisp Native Am/AK Native</td>
<td headers="SUD_MJ" class="gt_row gt_right" style="background-color: #FFFFE0;">1</td>
<td headers="count" class="gt_row gt_right" style="background-color: #FFFFE0;">164</td>
<td headers="percentage" class="gt_row gt_right" style="background-color: #FFFFE0;">27.3</td></tr>
    <tr><td headers="race" class="gt_row gt_center">NonHisp Native HI/Other Pac Isl</td>
<td headers="SUD_MJ" class="gt_row gt_right">0</td>
<td headers="count" class="gt_row gt_right">165</td>
<td headers="percentage" class="gt_row gt_right">84.6</td></tr>
    <tr><td headers="race" class="gt_row gt_center">NonHisp Native HI/Other Pac Isl</td>
<td headers="SUD_MJ" class="gt_row gt_right">1</td>
<td headers="count" class="gt_row gt_right">30</td>
<td headers="percentage" class="gt_row gt_right">15.4</td></tr>
    <tr><td headers="race" class="gt_row gt_center">NonHisp Asian</td>
<td headers="SUD_MJ" class="gt_row gt_right">0</td>
<td headers="count" class="gt_row gt_right">2301</td>
<td headers="percentage" class="gt_row gt_right">94.8</td></tr>
    <tr><td headers="race" class="gt_row gt_center" style="background-color: #F0FFF0;">NonHisp Asian</td>
<td headers="SUD_MJ" class="gt_row gt_right" style="background-color: #F0FFF0;">1</td>
<td headers="count" class="gt_row gt_right" style="background-color: #F0FFF0;">127</td>
<td headers="percentage" class="gt_row gt_right" style="background-color: #F0FFF0;">5.2</td></tr>
    <tr><td headers="race" class="gt_row gt_center">NonHisp more than one race</td>
<td headers="SUD_MJ" class="gt_row gt_right">0</td>
<td headers="count" class="gt_row gt_right">1297</td>
<td headers="percentage" class="gt_row gt_right">76.9</td></tr>
    <tr><td headers="race" class="gt_row gt_center">NonHisp more than one race</td>
<td headers="SUD_MJ" class="gt_row gt_right">1</td>
<td headers="count" class="gt_row gt_right">390</td>
<td headers="percentage" class="gt_row gt_right">23.1</td></tr>
    <tr><td headers="race" class="gt_row gt_center">Hispanic</td>
<td headers="SUD_MJ" class="gt_row gt_right">0</td>
<td headers="count" class="gt_row gt_right">6584</td>
<td headers="percentage" class="gt_row gt_right">84.9</td></tr>
    <tr><td headers="race" class="gt_row gt_center">Hispanic</td>
<td headers="SUD_MJ" class="gt_row gt_right">1</td>
<td headers="count" class="gt_row gt_right">1167</td>
<td headers="percentage" class="gt_row gt_right">15.1</td></tr>
  </tbody>
  
  
</table>
</div>
```

**Health**
The following bar plot shows that individuals with health problems have a higher percentage (20.7%) of SUD_MJ compared to those without health problems (13.6%). Please note that the survey data did not specify the specific health problems for individuals who rate their health as fair/poor.

```r
# health (0=w/o health problem: excellent/very good/good, 1=with health problem: fair/poor)
health_percentage <- sud_2022 %>%
  group_by(health, SUD_MJ) %>%
  summarize(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)

health_percentage <- health_percentage %>%
  mutate(health = factor(health, levels = c(0, 1), labels = c("w/o health problem", "with health problem")),
         SUD_MJ = factor(SUD_MJ, levels = c(0, 1), labels = c("0", "1")))
p3 <- sud_2022 %>% ggplot(
  aes(x = factor(health, levels = 0:1, labels = c("w/o health problem", "with health problem")),
      fill = factor(SUD_MJ))) +
  geom_bar(alpha = 0.5, position = "dodge") +
  scale_fill_manual(values = c("#619CFF", "#FF595E"),name = "SUD_MJ") +
  labs(x = "Health", y = "Count") +
  theme_minimal()+
  geom_text(data = health_percentage, aes(x = health, y = count, label = paste0(round(percentage, 1), "%")),
            position = position_dodge(width = 0.9), vjust = -0.5)
p3
```

![](Final_Writeup_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

**Marital**
The bar plot shows the distribution of marijuana use disorder (SUD_MJ) among different marital statuses. The variable is categorized into three groups: "Never married," "Married," and "Widowed/Divorced." The percentages of individuals with and without SUD_MJ are displayed on the bars within each group.

- Among those who have never married, 21.9% have a marijuana use disorder, while 78.1% do not.
- For married individuals, 7.2% have a marijuana use disorder, while a substantial 92.8% do not.
- In the widowed/divorced group, 11.6% have a marijuana use disorder, whereas 88.4% do not.

This plot highlights that the highest percentage of marijuana use disorder is found among those who have never married, followed by the widowed/divorced group. Married individuals have the lowest percentage of marijuana use disorder.

```r
# marital (0=never been married/cannot married<=14, 1=married, 2=widowed/divorced/separated)
marital_percentage <- sud_2022 %>%
  group_by(marital, SUD_MJ) %>%
  summarize(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)
marital_percentage <- marital_percentage %>%
  mutate(marital = factor(marital, levels = c(0, 1, 2), labels = c("Never married", "Married", "Widowed/Divorced")),
         SUD_MJ = factor(SUD_MJ, levels = c(0, 1), labels = c("0", "1")))
p4 <- sud_2022 %>% ggplot(
  aes(x = factor(marital, levels = 0:2, labels = c("Never married", "Married", 
                                                   "Widowed/Divorced")), fill = factor(SUD_MJ))) +
  geom_bar(alpha = 0.5, position = "dodge") +
  scale_fill_manual(values = c("#619CFF", "#FF595E"),name = "SUD_MJ") +
  labs(x = "Marital", y = "Count") +
  theme_minimal() + 
  geom_text(data = marital_percentage, aes(x = marital, y = count, label = paste0(round(percentage, 1), "%")),
            position = position_dodge(width = 0.9), vjust = -0.5)
p4
```

![](Final_Writeup_files/figure-html/unnamed-chunk-10-1.png)<!-- -->

**Education**
The bar plot on the left displays the distribution of marijuana use disorder (SUD_MJ) among individuals with different educational levels. The variable "degree" shows the highest degree repsondents obtained, and is categorized into three groups: "without high school degree", "high school degree", and "higher (associate's degree/college graduate or higher)". The percentages of individuals with and without SUD_MJ within each educational group are displayed on the bars.

- Among individuals without a high school degree, 18.6% have a marijuana use disorder, while 81.4% do not.
- For individuals with a high school degree, 18.2% have a marijuana use disorder, while 81.8% do not.
- Among individuals with higher education, 9.7% have a marijuana use disorder, whereas 90.3% do not.

In summary, the plot on the left highlights that individuals with higher education have the lowest percentage (9.7%) of marijuana use disorder, whereas individuals without a high school degree and those with a high school degree have similar higher percentages (18.6% and 18.2%, respectively)

I took a closer look at the distribution of marijuana use disorder (SUD_MJ) among students currently attending school compared to those who are not. The data reveals that a significant majority of individuals not currently attending school do not have a marijuana use disorder, with the 'No' group predominantly represented by individuals without SUD_MJ.

```r
# degree (1=w/o high school, 2=high school degree, 3=associate's degree/college graduate or higher)
educ_percentage <- sud_2022 %>%
  group_by(degree, SUD_MJ) %>%
  summarize(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)
educ_percentage <- educ_percentage %>%
  mutate(marital = factor(degree, levels = c(1, 2, 3), labels = c("w/o high school", "High school", "Higher")),
         SUD_MJ = factor(SUD_MJ, levels = c(0, 1), labels = c("0", "1")))
p5 <- sud_2022 %>% ggplot(
  aes(x = factor(degree, levels = 1:3, labels = c("w/o high school", "High school", 
                                                   "Higher")), fill = factor(SUD_MJ))) +
  geom_bar(alpha = 0.5, position = "dodge") +
  scale_fill_manual(values = c("#619CFF", "#FF595E"),name = "SUD_MJ") +
  labs(x = "Degree", y = "Count") +
  theme_minimal()+
  geom_text(data = educ_percentage, aes(x = marital, y = count, label = paste0(round(percentage, 1), "%")),
            position = position_dodge(width = 0.9), vjust = -0.5)
p5
```

![](Final_Writeup_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

```r
#  Now going to school or not? (1/11 = now going to school,  0=No, other is NA)
p5.1 <- sud_2022 %>%
  ggplot(aes(x = factor(student, levels = c(0, 1), labels = c("No", "Yes")), fill = factor(student))) +
  geom_bar(alpha = 0.7) +
  scale_fill_manual(values = c("#619CFF", "#FF595E"),name = "SUD_MJ") + 
  labs(x = "Now Going to School", y = "Count") +
  theme_minimal()
p5.1
```

![](Final_Writeup_files/figure-html/unnamed-chunk-11-2.png)<!-- -->

```r
grid.arrange(p5, p5.1, ncol = 2, widths = c(2, 1))
```

![](Final_Writeup_files/figure-html/unnamed-chunk-11-3.png)<!-- -->

**Employment**
This bar plot illustrates the distribution of marijuana use disorder (SUD_MJ) among individuals with different employment statuses: "Full time," "Part time," "Unemployed," and "Other." 

- Among individuals employed full-time, 14.1% have a marijuana use disorder, while 85.9% do not.
- For those employed part-time, 16.8% have a marijuana use disorder, while 83.2% do not.
- Among unemployed individuals, 20.1% have a marijuana use disorder, whereas 79.9% do not.
- In the "Other" employment category, 9.2% have a marijuana use disorder, while 90.8% do not.

This plot highlights that the highest percentage of marijuana use disorder is found among unemployed individuals (20.1%), followed by those employed part-time (16.8%). Individuals employed full-time and those in the "Other" employment category have lower percentages of marijuana use disorder, with the "Other" category having the lowest percentage (9.2%).

```r
# employ (1=employed full time, 2=employed part time, 3=unemployed, 4=Other(incl. not in labor force))
employ_percentage <- sud_2022 %>%
  group_by(employ, SUD_MJ) %>%
  summarize(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)
employ_percentage <- employ_percentage %>%
  mutate(marital = factor(employ, levels = c(1, 2, 3, 4), labels = c("Full time", "Part time", 
                                                   "Unemployed", "Other")),
         SUD_MJ = factor(SUD_MJ, levels = c(0, 1), labels = c("0", "1")))
p6 <- sud_2022 %>% ggplot(
  aes(x = factor(employ, levels = 1:4, labels = c("Full time", "Part time", 
                                                   "Unemployed", "Other")), 
      fill = factor(SUD_MJ))) +
  geom_bar(alpha = 0.5, position = "dodge") +
  scale_fill_manual(values = c("#619CFF", "#FF595E"),name = "SUD_MJ") +
  labs(x = "Employ", y = "Count") +
  theme_minimal()+
  geom_text(data = employ_percentage, aes(x = employ, y = count, label = paste0(round(percentage, 1), "%")),
            position = position_dodge(width = 0.9), vjust = -0.5)
p6
```

![](Final_Writeup_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

**Income**
This bar plot illustrates the distribution of marijuana use disorder (SUD_MJ) among individuals with different family income levels: "Poverty," "Middle," and "Wealth." 

- Among individuals in the "Poverty" income group, 20.9% have a marijuana use disorder, while 79.1% do not.
- For individuals in the "Middle" income group, 16.2% have a marijuana use disorder, while 83.8% do not.
- Among individuals in the "Wealth" income group, 10.1% have a marijuana use disorder, whereas 89.9% do not.

The result highlights that the highest percentage of marijuana use disorder is found among individuals in the "Poverty" income group (20.9%), followed by those in the "Middle" income group (16.2%). Individuals in the "Wealth" income group have the lowest percentage of marijuana use disorder (10.1%).

```r
income_percentage <- sud_2022 %>%
  group_by(income, SUD_MJ) %>%
  summarize(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)
income_percentage <- income_percentage %>%
  mutate(marital = factor(income, levels = c(1, 2, 3), labels = c("Poverty", "Middle", "Wealth")),
         SUD_MJ = factor(SUD_MJ, levels = c(0, 1), labels = c("0", "1")))
p9 <- sud_2022 %>% ggplot(
  aes(x = factor(income, levels = 1:3, labels = c("Poverty", "Middle", "Wealth")), fill = factor(SUD_MJ))) +
  geom_bar(alpha = 0.5, position = "dodge") +
  scale_fill_manual(values = c("#619CFF", "#FF595E"),name = "SUD_MJ") +
  labs(x = "Family Income", y = "Count") +
  theme_minimal()+
  geom_text(data = income_percentage, aes(x = income, y = count, label = paste0(round(percentage, 1), "%")),
            position = position_dodge(width = 0.9), vjust = -0.5)
p9
```

![](Final_Writeup_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

**Health Insurance**
This last bar plot illustrates the distribution of marijuana use disorder (SUD_MJ) among individuals with and without health insurance. The percentages of individuals with and without SUD_MJ within each health insurance group are displayed on the bars.

- Among individuals without health insurance, 19.6% have a marijuana use disorder, while 80.4% do not.
- For individuals with health insurance, 13.9% have a marijuana use disorder, whereas 86.1% do not.

The result highlights that the highest percentage of marijuana use disorder is found among individuals without health insurance (19.6%). In contrast, those with health insurance have a lower percentage of marijuana use disorder (13.9%).

```r
# health_insur (0=w/o health insurance, 1=health insurance)
insur_percentage <- sud_2022 %>%
  group_by(health_insur, SUD_MJ) %>%
  summarize(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)

insur_percentage <- insur_percentage %>%
  mutate(health_insur = factor(health_insur, levels = c(0, 1), labels = c("w/o insurance", "insurance")),
         SUD_MJ = factor(SUD_MJ, levels = c(0, 1), labels = c("0", "1")))

p10 <- sud_2022 %>% ggplot(
  aes(x = factor(health_insur, levels = c(0, 1), labels = c("w/o insurance", "insurance")), fill = factor(SUD_MJ))) +
  geom_bar(alpha = 0.5, position = "dodge") +
  scale_fill_manual(values = c("#619CFF", "#FF595E"),name = "SUD_MJ") +
  labs(x = "Health Insurance", y = "Count") +
  theme_minimal()+
  geom_text(data = insur_percentage, aes(x = health_insur, y = count, label = paste0(round(percentage, 1), "%")),
            position = position_dodge(width = 0.9), vjust = -0.5)
p10
```

![](Final_Writeup_files/figure-html/unnamed-chunk-14-1.png)<!-- -->

### Hypothesis Testing

In this section, I performed Chi-squared tests for all categorical variables against the marijuana use disorder (SUD_MJ) variable to test for independence. All variables in the sud_2022 dataset are converted to factors to ensure they are treated as categorical. For each categorical variable, a Chi-squared test of independence is conducted against SUD_MJ.

The Chi-squared test evaluates whether there is a significant association between each categorical variable and SUD_MJ. A p-value is obtained from each test, and variables with p-values less than 0.05 are considered significantly associated with SUD_MJ. 

The table shows that all tested variables (age, sex, race, health, marital status, degree, student status, employment, family, and presence of children) have p-values less than 0.05, indicating significant associations with SUD_MJ. However, the Chi-squared tests performed in this step only assess the independence of each individual categorical variable with respect to the marijuana use disorder (SUD_MJ) variable. Note that these tests do not provide information about the correlations or associations between the predictor variables themselves.


```r
sud_2022 <- sud_2022 %>%
  mutate_all(as.factor)
cate_var <- sud_2022 %>% select(-c(SUD_MJ))
variables <- names(cate_var)

# Perform Chi-squared Test
chi_square_test <- function(data, var) {
  tbl <- table(data[[var]], data[['SUD_MJ']])
  test <- chisq.test(tbl)
  p_value <- test$p.value
  data.frame(
    Variable = var,
    P_Value = p_value
  )
}

results <- lapply(variables, function(var) {
  chi_square_test(sud_2022, var)
})

# Combine the results 
results_df <- do.call(rbind, results)
results_df
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":["Variable"],"name":[1],"type":["chr"],"align":["left"]},{"label":["P_Value"],"name":[2],"type":["dbl"],"align":["right"]}],"data":[{"1":"age","2":"0.000000e+00"},{"1":"sex","2":"6.829754e-42"},{"1":"race","2":"6.762415e-86"},{"1":"health","2":"9.195597e-45"},{"1":"marital","2":"0.000000e+00"},{"1":"degree","2":"7.890639e-145"},{"1":"student","2":"8.308763e-06"},{"1":"employ","2":"2.458101e-78"},{"1":"family","2":"9.327843e-03"},{"1":"kid","2":"1.260493e-16"},{"1":"elderly","2":"2.483698e-89"},{"1":"health_insur","2":"1.075480e-24"},{"1":"income","2":"3.768875e-128"},{"1":"mentalhealth","2":"0.000000e+00"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

# Model Fitting

In the model fitting section, I use the cleaned dataset sud_2022 with selected predictors. Before applying the machine learning classification models, I first examined the outcome variable, marijuana use disorder (SUD_MJ). The outcome shows that our data is highly imbalanced, with 15% of individuals having SUD_MJ = 'Yes' and 85% having SUD_MJ = 'No'. This imbalance needs to be addressed to ensure that the classification models perform effectively and do not bias towards the majority class. 

I split the dataset into 67% for training and 33% for testing. To avoid overfitting, I initiated a 10-fold cross-validation that can be used later in the models.


```r
# Load in dataset
NSDUH_2022 <- read.csv("/Users/jiashuliu/Desktop/Projects/substance_use_disorder/data/sud_2022.csv")
# Convert all the variables into factors
NSDUH_2022 <- NSDUH_2022 %>% 
  mutate(across(where(is.numeric), as.factor))
str(NSDUH_2022)
```

```
## 'data.frame':	45331 obs. of  15 variables:
##  $ age         : Factor w/ 3 levels "2","3","4": 1 2 1 1 2 2 2 1 1 2 ...
##  $ sex         : Factor w/ 2 levels "0","1": 2 1 1 2 1 2 1 1 2 1 ...
##  $ race        : Factor w/ 7 levels "1","2","3","4",..: 7 2 1 5 1 1 1 1 1 3 ...
##  $ health      : Factor w/ 2 levels "0","1": 1 1 1 1 1 2 1 1 2 1 ...
##  $ marital     : Factor w/ 3 levels "0","1","2": 2 1 2 2 3 3 2 1 1 2 ...
##  $ degree      : Factor w/ 3 levels "1","2","3": 1 3 3 2 3 1 3 1 3 1 ...
##  $ student     : Factor w/ 2 levels "0","1": 1 1 2 1 1 1 1 1 1 1 ...
##  $ employ      : Factor w/ 4 levels "1","2","3","4": 3 1 1 1 1 3 1 3 3 4 ...
##  $ family      : Factor w/ 6 levels "1","2","3","4",..: 3 3 2 6 3 4 6 2 2 2 ...
##  $ kid         : Factor w/ 4 levels "0","1","2","3": 1 3 1 2 2 1 4 1 1 1 ...
##  $ elderly     : Factor w/ 3 levels "0","1","2": 1 1 1 1 1 1 1 1 1 1 ...
##  $ health_insur: Factor w/ 2 levels "0","1": 2 2 2 2 2 2 2 2 2 2 ...
##  $ income      : Factor w/ 3 levels "1","2","3": 2 2 3 3 2 2 3 2 2 2 ...
##  $ mentalhealth: Factor w/ 25 levels "0","1","2","3",..: 1 2 9 4 7 4 2 8 9 9 ...
##  $ SUD_MJ      : Factor w/ 2 levels "0","1": 1 2 1 1 1 2 1 2 2 1 ...
```

```r
# NSDUH_2022 is an imbalanced dataset
table(NSDUH_2022$SUD_MJ)
```

```
## 
##     0     1 
## 38754  6577
```

```r
sud_mj_yes <- subset(NSDUH_2022, SUD_MJ == 1)
sud_mj_no <- subset(NSDUH_2022, SUD_MJ == 0)
ratio_yes <- nrow(sud_mj_yes) / nrow(NSDUH_2022)
ratio_no <- nrow(sud_mj_no) / nrow(NSDUH_2022)

# Display the ratios
cat("Ratio of SUD_MJ = Yes:", round(ratio_yes, 2), "\n")
```

```
## Ratio of SUD_MJ = Yes: 0.15
```

```r
cat("Ratio of SUD_MJ = No:", round(ratio_no, 2), "\n")
```

```
## Ratio of SUD_MJ = No: 0.85
```

```r
# Split training and Testing
set.seed(123)
# find split_size to divide data in 67% train/ 33% test sets
split_size <- sample(1:nrow(NSDUH_2022), floor(0.67 * nrow(NSDUH_2022)))

# Extract the train and test sets
train <- NSDUH_2022[split_size, ]
test <- NSDUH_2022[-split_size, ]
# Change the levels of SUD_MJ from 0 and 1 to 'No' and 'Yes'.
# Otherwise this will lead to errors in trainControl
levels(train$SUD_MJ) <- c("No", "Yes")
levels(test$SUD_MJ) <- c("No", "Yes")
#levels(train$SUD_MJ)
#levels(test$SUD_MJ)
# Stratified cross-validation
trControl <- trainControl(method = "cv", 
                          number = 10, 
                          classProbs = TRUE, 
                          summaryFunction = twoClassSummary)
```

### 1. Logistic Regression

I first start with modeling using logistic regression, as it is one of the most straightforward and interpretable models that is suitable for binary classification tasks. The logistic regression model allows us to estimate the probability that a given input belongs to a specific class (in this case, whether an individual has marijuana use disorder). It could serve as a good baseline model against which more complex models can be compared.


```r
# Train logistic regression model on training set
set.seed(123)
logistic_cv <- caret::train(SUD_MJ ~ ., 
                                  data = train, 
                                  method = "glm", 
                                  family = binomial, 
                                  trControl = trControl, # using cv to avoid overfitting
                                  metric = "ROC")
# print(logistic_cv)

# Make predictions on the testing set
test$predicted_prob_logistic <- predict(logistic_cv, newdata = test, type = "prob")[, "Yes"]
test$predicted_class_logistic <- ifelse(test$predicted_prob_logistic > 0.5, "Yes", "No")
test$predicted_class_logistic <- factor(test$predicted_class_logistic, levels = c("No", "Yes"))
```

#### Model Performance 

After fitting the logistic regression model to the training data and making predictions on the test data, I evaluated the model's performance using a confusion matrix and various performance metrics.

- Accuracy: The model achieved an accuracy of 85.26%, indicating that in general, 85.26% of the predictions made by the model were correct.
- Precision: The precision, or positive predictive value, is 85.78%, meaning that 85.78% of the individuals predicted to not have SUD_MJ actually do not have it.
- Specificity: The specificity in this model is very low at 3.93%, indicating that the model struggled to correctly identify individuals with SUD_MJ.
- Recall: The recall rate, also called sensitivity, is 99.19%, meaning that the model correctly identified 99.19% of the individuals who do not have SUD_MJ.
- F1-Score: The F1-score is 92%, providing a balance between precision and recall.
- AUC: The AUC is 0.76, indicating that the model has a fair ability to distinguish between the two classes.

The model's high sensitivity but low specificity highlight the challenges of correctly identifying respondents with SUD_MJ with this imbalanced dataset. In the context of an imbalanced dataset, traditional metrics like accuracy can be misleading because they are dominated by the majority class. AUC (Area Under the Curve) is a more reliable measure in this case , as it measures the ability of the model to discriminate between the positive and negative classes across different threshold settings. An AUC of 0.76 suggests that the model has a fair ability to distinguish between individuals with and without SUD_MJ.


```r
confusion_matrix_logistic <- confusionMatrix(test$predicted_class_logistic, test$SUD_MJ)
print(confusion_matrix_logistic)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    No   Yes
##        No  12669  2101
##        Yes   104    86
##                                           
##                Accuracy : 0.8526          
##                  95% CI : (0.8468, 0.8583)
##     No Information Rate : 0.8538          
##     P-Value [Acc > NIR] : 0.6665          
##                                           
##                   Kappa : 0.0502          
##                                           
##  Mcnemar's Test P-Value : <2e-16          
##                                           
##             Sensitivity : 0.99186         
##             Specificity : 0.03932         
##          Pos Pred Value : 0.85775         
##          Neg Pred Value : 0.45263         
##              Prevalence : 0.85381         
##          Detection Rate : 0.84686         
##    Detection Prevalence : 0.98730         
##       Balanced Accuracy : 0.51559         
##                                           
##        'Positive' Class : No              
## 
```

```r
# accuracy
accuracy_logi <- confusion_matrix_logistic$overall['Accuracy']
print(paste("Accuracy:", round(accuracy_logi, 2)))
```

```
## [1] "Accuracy: 0.85"
```

```r
# Precision, Recall, F1-Score
precision_logi <- confusion_matrix_logistic$byClass['Pos Pred Value']
recall_logi <- confusion_matrix_logistic$byClass['Sensitivity'] 
F1_score_logi <- 2 * ((precision_logi * recall_logi) / (precision_logi + recall_logi))
print(paste("Precision:", round(precision_logi, 2)))
```

```
## [1] "Precision: 0.86"
```

```r
print(paste("Recall:", round(recall_logi, 2)))
```

```
## [1] "Recall: 0.99"
```

```r
print(paste("F1-Score:", round(F1_score_logi, 2)))
```

```
## [1] "F1-Score: 0.92"
```

```r
# AUC
pred <- prediction(test$predicted_prob, test$SUD_MJ)
perf_auc <- performance(pred, measure = "auc")
auc_value <- perf_auc@y.values[[1]]
print(paste("AUC:", round(auc_value, 2)))
```

```
## [1] "AUC: 0.76"
```

```r
perf_roc <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf_roc, main = "ROC Curve", col = "red", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
```

![](Final_Writeup_files/figure-html/unnamed-chunk-18-1.png)<!-- -->

### 2. Random Forest

I further employed the random forest model to tackle the binary classification problem of predicting marijuana use disorder (SUD_MJ). Unlike logistic regression, which is a linear model, the random forest is a non-linear model capable of capturing more complex relationships between features. Additionally, the random forest model provides feature importance metrics, which help identify the most relevant features for the classification task.

The model was initially trained on the training data with 500 trees, and the number of variables tried at each split (mtry) was set to the optimal value found during tuning. The optimal mtry value was determined by tuning the model to minimize the Out-Of-Bag (OOB) error estimate. The OOB error estimate serves as an internal validation metric for the model, providing an unbiased estimate of the prediction error. The random forest model achieved a minimum OOB error rate of 14.48%, indicating that approximately 14.48% of the predictions made by the model on the training data were incorrect. I selected the mtry value that corresponded to the lowest OOB error rate for the final model.


```r
# Train random forest model using training data 
set.seed(123)
rf_model<- randomForest(SUD_MJ ~ ., 
                        data = train, 
                        ntree = 500, 
                        importance = TRUE)
# Tune the Rf model by finding the optimal mtry value 
# Select mtry value with minimum out of bag(OOB) error
mtry <- tuneRF(train[, -ncol(train)],train$SUD_MJ, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
```

```
## mtry = 3  OOB error = 14.7% 
## Searching left ...
## mtry = 2 	OOB error = 14.45% 
## 0.01724524 0.01 
## Searching right ...
## mtry = 4 	OOB error = 15.11% 
## -0.04603464 0.01
```

![](Final_Writeup_files/figure-html/unnamed-chunk-19-1.png)<!-- -->

```r
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
```

```
##       mtry  OOBError
## 2.OOB    2 0.1444799
## 3.OOB    3 0.1470152
## 4.OOB    4 0.1511310
```

```r
print(best.m)
```

```
## [1] 2
```

In the next step, I refit the model using the best mtry value (mtry = 2), generate variable importance plots, and make predictions using the testing data. The variable importance plot highlights the most influential predictors in the Random Forest model. According to the Mean Decrease Accuracy and Mean Decrease Gini metrics, 'mentalhealth', 'age', 'race', 'marital', 'family', and 'employ' were among the most important variables for predicting SUD_MJ. The Mean Decrease Accuracy measures the reduction in model accuracy when a variable is excluded, while the Mean Decrease Gini indicates the total decrease in node impurity that a variable contributes across all trees in the forest.


```r
# Refit the model using best mtry
rf_tuned <- randomForest(SUD_MJ ~., data = train, ntree = 500, mtry = best.m, importance=TRUE)
print(rf_tuned)
```

```
## 
## Call:
##  randomForest(formula = SUD_MJ ~ ., data = train, ntree = 500,      mtry = best.m, importance = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 14.46%
## Confusion matrix:
##        No Yes class.error
## No  25969  12 0.000461876
## Yes  4380  10 0.997722096
```

```r
importance(rf_tuned) 
```

```
##                    No         Yes MeanDecreaseAccuracy MeanDecreaseGini
## age          31.62330  -6.8097770             33.69835        146.91830
## sex          16.74189   1.1962322             16.26212         78.40928
## race         16.05316  12.0027117             20.94219        239.61525
## health       17.51914   7.5985078             20.50821         64.80729
## marital      33.60580 -11.0479339             34.99733        186.22278
## degree       19.76163   6.9917209             24.51384        138.35584
## student      28.71188 -15.8802242             24.43010         64.41979
## employ       34.21988 -10.8155253             30.61820        168.58646
## family       29.49790  -9.4380882             27.59184        228.11015
## kid          28.35491 -14.0350027             25.61441        131.15952
## elderly      30.19227 -17.1138738             27.86301         76.28111
## health_insur 11.14520   8.5301862             14.83579         56.18484
## income       22.01537  -0.2403833             23.17022        131.46384
## mentalhealth 35.57704  52.8341071             62.84015        820.61047
```

```r
varImpPlot(rf_tuned)
```

![](Final_Writeup_files/figure-html/unnamed-chunk-20-1.png)<!-- -->

```r
# Higher the value of mean decrease accuracy or mean decrease gini score , higher the importance of the variable in the model. In the plot shown blow, mental health is most important variable.
# Mean Decrease Accuracy - How much the model accuracy decreases if we drop that variable.
# Mean Decrease Gini - Measure of variable importance based on the Gini impurity index used for the calculation of splits in trees.
# Making predictions based on test data
test$pred_prob_rf_tuned <- predict(rf_tuned, newdata = test, type = "prob")[, 2]
test$pred_class_rf_tuned <- ifelse(test$pred_prob_rf_tuned > 0.5, "Yes", "No")
# Print confusion matrix
confusion_matrix_rf_tuned <- confusionMatrix(as.factor(test$pred_class_rf_tuned), test$SUD_MJ)
```

#### Model Performance

- Accuracy: Evaluating on the testing data, the Random Forest model achieved an accuracy of 85.41%, with a 95% confidence interval ranging from 84.83% to 85.97%.
- Precision: The precision is 85.41%, meaning that when the model predicted "No" for SUD_MJ, it was correct 85.41% of the time.
- Recall (sensitivity): In this model, the recall rate is extremely high at 99.99%, indicating that the model correctly identified nearly all individuals without SUD_MJ.
- Specificity: The specificity is very low at 0.23%, reflecting the model's difficulty in correctly identifying individuals with SUD_MJ. 
- F1-Score: The F1-Score is 92%, providing a balance between precision and recall.
- AUC: The AUC for the random forest model is 0.72, which is even lower than that of the logistic regression model. 


```r
# Accuracy
print(confusion_matrix_rf_tuned)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    No   Yes
##        No  12771  2182
##        Yes     2     5
##                                           
##                Accuracy : 0.854           
##                  95% CI : (0.8483, 0.8596)
##     No Information Rate : 0.8538          
##     P-Value [Acc > NIR] : 0.478           
##                                           
##                   Kappa : 0.0036          
##                                           
##  Mcnemar's Test P-Value : <2e-16          
##                                           
##             Sensitivity : 0.999843        
##             Specificity : 0.002286        
##          Pos Pred Value : 0.854076        
##          Neg Pred Value : 0.714286        
##              Prevalence : 0.853810        
##          Detection Rate : 0.853676        
##    Detection Prevalence : 0.999532        
##       Balanced Accuracy : 0.501065        
##                                           
##        'Positive' Class : No              
## 
```

```r
accuracy_rf_tuned <- confusion_matrix_rf_tuned$overall['Accuracy']
print(paste("Accuracy:", round(accuracy_rf_tuned, 2)))
```

```
## [1] "Accuracy: 0.85"
```

```r
# Precision, Recall, F1-Score
precision_rf_tuned <- confusion_matrix_rf_tuned$byClass['Pos Pred Value']
recall_rf_tuned <- confusion_matrix_rf_tuned$byClass['Sensitivity']
F1_rf_tuned <- 2 * ((precision_rf_tuned * recall_rf_tuned) / (precision_rf_tuned + recall_rf_tuned))
print(paste("Precision:", round(precision_rf_tuned, 2)))
```

```
## [1] "Precision: 0.85"
```

```r
print(paste("Recall:", round(recall_rf_tuned, 2)))
```

```
## [1] "Recall: 1"
```

```r
print(paste("F1-Score:", round(F1_rf_tuned, 2)))
```

```
## [1] "F1-Score: 0.92"
```

```r
# AUC
pred_rf_tuned <- prediction(test$pred_prob_rf_tuned, test$SUD_MJ)
auc_rf_tuned <- performance(pred_rf_tuned, "auc")
auc_value_rf_tuned <- auc_rf_tuned@y.values[[1]]
print(paste("AUC:", round(auc_value_rf_tuned, 2)))
```

```
## [1] "AUC: 0.72"
```

```r
roc_rf_tuned <- performance(pred_rf_tuned, measure = "tpr", x.measure = "fpr")
plot(roc_rf_tuned, main="ROC Curve for Random Forest",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
```

![](Final_Writeup_files/figure-html/unnamed-chunk-21-1.png)<!-- -->

### 3. Gradient Boosting Model

```r
# One-hot encode using model.matrix
train_matrix <- model.matrix(SUD_MJ ~ . - 1, data = train)
test_matrix <- model.matrix(SUD_MJ ~ . - 1, data = test)

# Convert the results to data frames
train_encoded <- as.data.frame(train_matrix)
test_encoded <- as.data.frame(test_matrix)

# Ensure the target variable SUD_MJ is included
train_encoded$SUD_MJ <- train$SUD_MJ
test_encoded$SUD_MJ <- test$SUD_MJ

# Identify numeric columns
numeric_cols_train <- sapply(train_encoded, is.numeric)
numeric_cols_test <- sapply(test_encoded, is.numeric)

# Apply normalization (scaling)
train_encoded[numeric_cols_train] <- scale(train_encoded[numeric_cols_train])
test_encoded[numeric_cols_test] <- scale(test_encoded[numeric_cols_test])

# Verify column names again after scaling
identical(names(train_encoded), names(test_encoded))
```

```
## [1] FALSE
```


```r
# Set the seed for reproducibility
set.seed(123)
# 10-folds cross-validation to prevent overfitting
trControl <- trainControl(method = "cv", 
                          number = 10, 
                          classProbs = TRUE, 
                          summaryFunction = twoClassSummary,
                          search = "grid")
# tune the hyperparameters
tuneGrid <- expand.grid(interaction.depth = c(1, 3, 5),  # Depth of each tree
                        n.trees = c(50, 100, 150),       # Number of trees
                        shrinkage = c(0.01, 0.1, 0.3),   # Learning rate
                        n.minobsinnode = c(10, 20))
# fit gbm model with train data
gbm_model <- caret::train(SUD_MJ ~ ., 
                          data = train_encoded, 
                          method = "gbm", 
                          trControl = trControl, 
                          tuneGrid = tuneGrid, 
                          metric = "ROC", 
                          verbose = FALSE)
print(gbm_model)
```

```
## Stochastic Gradient Boosting 
## 
## 30371 samples
##    56 predictor
##     2 classes: 'No', 'Yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 27333, 27334, 27334, 27334, 27334, 27334, ... 
## Resampling results across tuning parameters:
## 
##   shrinkage  interaction.depth  n.minobsinnode  n.trees  ROC        Sens     
##   0.01       1                  10               50      0.6606615  1.0000000
##   0.01       1                  10              100      0.6775371  1.0000000
##   0.01       1                  10              150      0.6894952  1.0000000
##   0.01       1                  20               50      0.6610557  1.0000000
##   0.01       1                  20              100      0.6789322  1.0000000
##   0.01       1                  20              150      0.6884493  1.0000000
##   0.01       3                  10               50      0.7043603  1.0000000
##   0.01       3                  10              100      0.7100773  1.0000000
##   0.01       3                  10              150      0.7131742  1.0000000
##   0.01       3                  20               50      0.7040545  1.0000000
##   0.01       3                  20              100      0.7097042  1.0000000
##   0.01       3                  20              150      0.7132267  1.0000000
##   0.01       5                  10               50      0.7101017  1.0000000
##   0.01       5                  10              100      0.7157797  1.0000000
##   0.01       5                  10              150      0.7191753  1.0000000
##   0.01       5                  20               50      0.7096339  1.0000000
##   0.01       5                  20              100      0.7154154  1.0000000
##   0.01       5                  20              150      0.7191780  1.0000000
##   0.10       1                  10               50      0.7136552  1.0000000
##   0.10       1                  10              100      0.7297231  1.0000000
##   0.10       1                  10              150      0.7382853  0.9999230
##   0.10       1                  20               50      0.7147003  1.0000000
##   0.10       1                  20              100      0.7297452  0.9999615
##   0.10       1                  20              150      0.7394520  0.9999230
##   0.10       3                  10               50      0.7357178  0.9998075
##   0.10       3                  10              100      0.7475526  0.9979600
##   0.10       3                  10              150      0.7516516  0.9950348
##   0.10       3                  20               50      0.7356028  0.9998075
##   0.10       3                  20              100      0.7476589  0.9976521
##   0.10       3                  20              150      0.7515993  0.9950734
##   0.10       5                  10               50      0.7431635  0.9987299
##   0.10       5                  10              100      0.7507682  0.9945345
##   0.10       5                  10              150      0.7527424  0.9918402
##   0.10       5                  20               50      0.7433164  0.9987683
##   0.10       5                  20              100      0.7505973  0.9940341
##   0.10       5                  20              150      0.7521612  0.9916092
##   0.30       1                  10               50      0.7387339  0.9997691
##   0.30       1                  10              100      0.7501736  0.9973827
##   0.30       1                  10              150      0.7533193  0.9944189
##   0.30       1                  20               50      0.7381452  0.9998460
##   0.30       1                  20              100      0.7496742  0.9979215
##   0.30       1                  20              150      0.7532913  0.9947269
##   0.30       3                  10               50      0.7506859  0.9932643
##   0.30       3                  10              100      0.7516462  0.9888380
##   0.30       3                  10              150      0.7514273  0.9866825
##   0.30       3                  20               50      0.7504954  0.9937262
##   0.30       3                  20              100      0.7517616  0.9898388
##   0.30       3                  20              150      0.7510277  0.9875294
##   0.30       5                  10               50      0.7490036  0.9893385
##   0.30       5                  10              100      0.7486196  0.9844502
##   0.30       5                  10              150      0.7461652  0.9827181
##   0.30       5                  20               50      0.7511328  0.9898773
##   0.30       5                  20              100      0.7502270  0.9868365
##   0.30       5                  20              150      0.7482429  0.9846042
##   Spec        
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0000000000
##   0.0006833713
##   0.0000000000
##   0.0000000000
##   0.0004555809
##   0.0009111617
##   0.0118451025
##   0.0232346241
##   0.0011389522
##   0.0109339408
##   0.0259681093
##   0.0068337130
##   0.0268792711
##   0.0416856492
##   0.0082004556
##   0.0296127563
##   0.0410022779
##   0.0009111617
##   0.0136674260
##   0.0271070615
##   0.0009111617
##   0.0109339408
##   0.0266514806
##   0.0328018223
##   0.0530751708
##   0.0587699317
##   0.0264236902
##   0.0453302961
##   0.0569476082
##   0.0507972665
##   0.0617312073
##   0.0710706150
##   0.0503416856
##   0.0660592255
##   0.0706150342
## 
## ROC was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150, interaction.depth =
##  1, shrinkage = 0.3 and n.minobsinnode = 10.
```


```r
best_params <- gbm_model$bestTune
print(best_params)
```

```
##    n.trees interaction.depth shrinkage n.minobsinnode
## 39     150                 1       0.3             10
```


```r
# Convert target variable to numeric (0 and 1)
train_encoded$SUD_MJ <- ifelse(train_encoded$SUD_MJ == "Yes", 1, 0)
test_encoded$SUD_MJ <- ifelse(test_encoded$SUD_MJ == "Yes", 1, 0)

gbm_best <- gbm::gbm(SUD_MJ ~ ., 
                      data = train_encoded, 
                      distribution = "bernoulli", 
                      n.trees = best_params$n.trees, 
                      interaction.depth = best_params$interaction.depth, 
                      shrinkage = best_params$shrinkage, 
                      n.minobsinnode = best_params$n.minobsinnode, 
                      cv.folds = 10, 
                      keep.data = TRUE, 
                      verbose = FALSE)
summary(gbm_best)
```

![](Final_Writeup_files/figure-html/unnamed-chunk-25-1.png)<!-- --><div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["var"],"name":[1],"type":["chr"],"align":["left"]},{"label":["rel.inf"],"name":[2],"type":["dbl"],"align":["right"]}],"data":[{"1":"age2","2":"16.3186532","_rn_":"age2"},{"1":"marital1","2":"12.1920714","_rn_":"marital1"},{"1":"age4","2":"4.6200249","_rn_":"age4"},{"1":"degree3","2":"4.5242165","_rn_":"degree3"},{"1":"sex1","2":"3.2530688","_rn_":"sex1"},{"1":"mentalhealth12","2":"3.2152945","_rn_":"mentalhealth12"},{"1":"mentalhealth16","2":"3.2121709","_rn_":"mentalhealth16"},{"1":"mentalhealth24","2":"3.0395127","_rn_":"mentalhealth24"},{"1":"mentalhealth18","2":"2.9633906","_rn_":"mentalhealth18"},{"1":"mentalhealth14","2":"2.8545466","_rn_":"mentalhealth14"},{"1":"race5","2":"2.7746464","_rn_":"race5"},{"1":"mentalhealth20","2":"2.6907415","_rn_":"mentalhealth20"},{"1":"mentalhealth15","2":"2.5779868","_rn_":"mentalhealth15"},{"1":"health1","2":"2.5605541","_rn_":"health1"},{"1":"mentalhealth17","2":"2.5167799","_rn_":"mentalhealth17"},{"1":"student1","2":"2.4710658","_rn_":"student1"},{"1":"mentalhealth13","2":"2.2665946","_rn_":"mentalhealth13"},{"1":"mentalhealth9","2":"2.1260470","_rn_":"mentalhealth9"},{"1":"mentalhealth10","2":"2.1120305","_rn_":"mentalhealth10"},{"1":"income3","2":"2.0763541","_rn_":"income3"},{"1":"mentalhealth11","2":"1.9515746","_rn_":"mentalhealth11"},{"1":"mentalhealth22","2":"1.6168500","_rn_":"mentalhealth22"},{"1":"mentalhealth21","2":"1.6027575","_rn_":"mentalhealth21"},{"1":"mentalhealth19","2":"1.4875027","_rn_":"mentalhealth19"},{"1":"mentalhealth1","2":"1.4696651","_rn_":"mentalhealth1"},{"1":"race6","2":"1.3568273","_rn_":"race6"},{"1":"mentalhealth7","2":"1.2683412","_rn_":"mentalhealth7"},{"1":"mentalhealth8","2":"1.2458004","_rn_":"mentalhealth8"},{"1":"mentalhealth23","2":"1.2213087","_rn_":"mentalhealth23"},{"1":"race3","2":"1.1297726","_rn_":"race3"},{"1":"mentalhealth6","2":"0.8755914","_rn_":"mentalhealth6"},{"1":"family2","2":"0.8431360","_rn_":"family2"},{"1":"race2","2":"0.7549786","_rn_":"race2"},{"1":"kid3","2":"0.4569654","_rn_":"kid3"},{"1":"race7","2":"0.4331358","_rn_":"race7"},{"1":"employ4","2":"0.3091339","_rn_":"employ4"},{"1":"mentalhealth4","2":"0.2937680","_rn_":"mentalhealth4"},{"1":"mentalhealth2","2":"0.2901376","_rn_":"mentalhealth2"},{"1":"marital2","2":"0.2706359","_rn_":"marital2"},{"1":"family6","2":"0.2697946","_rn_":"family6"},{"1":"employ2","2":"0.2350592","_rn_":"employ2"},{"1":"mentalhealth5","2":"0.1296869","_rn_":"mentalhealth5"},{"1":"elderly1","2":"0.1218257","_rn_":"elderly1"},{"1":"age3","2":"0.0000000","_rn_":"age3"},{"1":"race4","2":"0.0000000","_rn_":"race4"},{"1":"degree2","2":"0.0000000","_rn_":"degree2"},{"1":"employ3","2":"0.0000000","_rn_":"employ3"},{"1":"family3","2":"0.0000000","_rn_":"family3"},{"1":"family4","2":"0.0000000","_rn_":"family4"},{"1":"family5","2":"0.0000000","_rn_":"family5"},{"1":"kid1","2":"0.0000000","_rn_":"kid1"},{"1":"kid2","2":"0.0000000","_rn_":"kid2"},{"1":"elderly2","2":"0.0000000","_rn_":"elderly2"},{"1":"health_insur1","2":"0.0000000","_rn_":"health_insur1"},{"1":"income2","2":"0.0000000","_rn_":"income2"},{"1":"mentalhealth3","2":"0.0000000","_rn_":"mentalhealth3"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

#### Model Performance

```r
# Make predictions on the testing set
test_encoded$predicted_prob_gbm <- predict(gbm_best, newdata = test_encoded, n.trees = best_params$n.trees, type = "response")
test_encoded$predicted_class_gbm <- ifelse(test_encoded$predicted_prob_gbm > 0.5, 1, 0)

# Confusion matrix to evaluate the GBM model on the testing set
confusion_matrix_test_gbm <- confusionMatrix(as.factor(test_encoded$predicted_class_gbm), as.factor(test_encoded$SUD_MJ))
print(confusion_matrix_test_gbm)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction     0     1
##          0 12719  2125
##          1    54    62
##                                         
##                Accuracy : 0.8543        
##                  95% CI : (0.8486, 0.86)
##     No Information Rate : 0.8538        
##     P-Value [Acc > NIR] : 0.4321        
##                                         
##                   Kappa : 0.0397        
##                                         
##  Mcnemar's Test P-Value : <2e-16        
##                                         
##             Sensitivity : 0.99577       
##             Specificity : 0.02835       
##          Pos Pred Value : 0.85684       
##          Neg Pred Value : 0.53448       
##              Prevalence : 0.85381       
##          Detection Rate : 0.85020       
##    Detection Prevalence : 0.99225       
##       Balanced Accuracy : 0.51206       
##                                         
##        'Positive' Class : 0             
## 
```

```r
# Calculate and print performance metrics for the GBM model on the testing set
accuracy_test_gbm <- confusion_matrix_test_gbm$overall['Accuracy']
print(paste("Accuracy:", round(accuracy_test_gbm, 2)))
```

```
## [1] "Accuracy: 0.85"
```

```r
precision_test_gbm <- confusion_matrix_test_gbm$byClass['Pos Pred Value']
recall_test_gbm <- confusion_matrix_test_gbm$byClass['Sensitivity']
f1_score_test_gbm <- 2 * ((precision_test_gbm * recall_test_gbm) / (precision_test_gbm + recall_test_gbm))

print(paste("Precision:", round(precision_test_gbm, 2)))
```

```
## [1] "Precision: 0.86"
```

```r
print(paste("Recall:", round(recall_test_gbm, 2)))
```

```
## [1] "Recall: 1"
```

```r
print(paste("F1-Score:", round(f1_score_test_gbm, 2)))
```

```
## [1] "F1-Score: 0.92"
```

```r
# Using ROCR to calculate AUC and plot ROC curve for the GBM model
pred_gbm <- prediction(test_encoded$predicted_prob_gbm, test_encoded$SUD_MJ)
perf_auc_gbm <- performance(pred_gbm, measure = "auc")
auc_value_gbm <- perf_auc_gbm@y.values[[1]]
print(paste("AUC:", round(auc_value_gbm, 2)))
```

```
## [1] "AUC: 0.76"
```

```r
# Performance object for ROC
perf_roc_gbm <- performance(pred_gbm, measure = "tpr", x.measure = "fpr")

# Plot the ROC curve for the GBM model
plot(perf_roc_gbm, main = "ROC Curve for GBM", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
```

![](Final_Writeup_files/figure-html/unnamed-chunk-26-1.png)<!-- -->

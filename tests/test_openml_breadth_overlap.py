"""The OpenML breadth suite must not overlap the benchmarks it is meant to steer towards.

`fintfm.experiments.openml_breadth.SUITE` is a decision loop: arms are chosen on it. Six of its
original fifteen datasets were TabArena-v0.1 datasets (design review 2026-10-16, task DR4.1),
so choosing on it biased the eventual TabArena number. This is evaluation, not training, so
decision D2 is untouched; the harm is selection bias, and nothing would warn of it.

The lists below are copied from the local TabArena fork (``~/proj/tabarena``, branch
``fintfm-audit-fixes`` at commit 2d8f7cfba072379587a78a820ec079d0e51fe3fa), because the test must
run without the fork or a network:

* TabArena-v0.1: ``packages/tabarena/src/tabarena/benchmark/task/metadata/data/
  curated_tabarena_dataset_metadata.csv``, columns ``dataset_id`` and ``dataset_name`` (51
  datasets; the same names as ``sources/data/TabArena-v0.1_tasks_metadata.csv``).
* BeyondArena: ``packages/tabarena/src/tabarena/benchmark/task/metadata/sources/data/
  BeyondArena_tasks_metadata.csv``, column ``dataset_name`` (142 datasets; no OpenML ids).

Not covered: a row-hash comparison, which needs every dataset's contents. TabArena's ids are its
curated re-uploads (46904 onwards), not the original OpenML uploads SUITE fetches (bank-marketing
is 1461 in SUITE and 46910 in TabArena), so the id check alone would have passed before the fix;
the name check is the one that caught the overlap.
"""

from __future__ import annotations

import re

from fintfm.experiments.openml_breadth import SUITE

#: TabArena-v0.1's 51 datasets as (OpenML dataset id, name). Source in the module docstring.
TABARENA_V0_1: tuple[tuple[int, str], ...] = (
    (46904, "airfoil_self_noise"), (46905, "Amazon_employee_access"), (46906, "anneal"),
    (46907, "Another-Dataset-on-used-Fiat-500"), (46908, "APSFailure"),
    (46910, "bank-marketing"), (46911, "Bank_Customer_Churn"), (46912, "Bioresponse"),
    (46913, "blood-transfusion-service-center"), (46915, "churn"),
    (46916, "coil2000_insurance_policies"), (46917, "concrete_compressive_strength"),
    (46918, "credit-g"), (46919, "credit_card_clients_default"),
    (46920, "customer_satisfaction_in_airline"), (46921, "diabetes"), (46922, "Diabetes130US"),
    (46923, "diamonds"), (46924, "E-CommereShippingData"), (46927, "Fitness_Club"),
    (46928, "Food_Delivery_Time"), (46929, "GiveMeSomeCredit"),
    (46930, "hazelnut-spread-contaminant-detection"), (46931, "healthcare_insurance_expenses"),
    (46932, "heloc"), (46933, "hiva_agnostic"), (46934, "houses"),
    (46935, "HR_Analytics_Job_Change_of_Data_Scientists"),
    (46937, "in_vehicle_coupon_recommendation"), (46938, "Is-this-a-good-customer"),
    (46939, "kddcup09_appetency"), (46940, "Marketing_Campaign"),
    (46941, "maternal_health_risk"), (46942, "miami_housing"),
    (46947, "online_shoppers_intention"), (46949, "physiochemical_protein"),
    (46950, "polish_companies_bankruptcy"), (46952, "qsar-biodeg"), (46953, "QSAR-TID-11"),
    (46954, "QSAR_fish_toxicity"), (46955, "SDSS17"), (46956, "seismic-bumps"),
    (46958, "splice"), (46960, "students_dropout_and_academic_success"),
    (46961, "superconductivity"), (46962, "taiwanese_bankruptcy_prediction"),
    (46963, "website_phishing"), (46964, "wine_quality"), (46969, "NATICUSdroid"),
    (46979, "jm1"), (46980, "MIC"),
)

#: BeyondArena's 142 dataset names, whitespace-separated. Source in the module docstring.
_BEYONDARENA_TEXT = """
5g_energy_consumption acquire_valued_shoppers_challenge airfoil_self_noise
allstate_claims_severity amazon_employee_access amex_non_iid_1m anes_voting_2026 aps_failure
asp_potassco_classification audiology_diagnosis bad_customer_detection bank_customer_churn
bank_marketing biogeographical_ancestry_prediction biomechanical_orthopaedic_prediction
bioresponse blood_tests_drink_prediction blood_transfusion body_density_prediction
california_house_prices_2020 cardiotocography churn cirrhosis_patient_survival_prediction
climate_model_weather_forecasting_1m clock_protein_toxicity coffee_rating_prediction coil_2000
concrete_compressive_strength consumer_complaints_1m cooking_time_1m covertype credit_approval
credit_card_clients_default credit_g customer_satisfaction_in_airline delivery_eta_1m
dementia_prediction diabetes_130_us diamonds drug_induced_autoimmunity_prediction
early_learning_predictors early_stage_diabetes_risk_prediction ecoli_proteins
ecommerce_shipping electric_motor_temperature_prediction emscad eryhemato_squamous_disease
fiat_500 fitness_club food_delivery_time forensic_glass_identification forest_fires
gallstone_disease garments_worker_productivity ghanas_indigenous_intel give_me_some_credit
hazelnut_spread_contaminant_detection healthcare_insurance_expenses heart_disease_cleveland
heart_disease_hungary heart_disease_va_long_beach heart_failure_followup_survival heloc
hepatitis_c_prediction hepatitis_survival_prediction hiva_agnostic home_credit_default_risk
home_credit_default_stability_1m homeq_default_prediction homesite_quote_conversion
horse_colic_survival hotel_booking_demand houses hr_analytics ieee_fraud_detection
immoscout_german_house_prices in_vehicle_coupon_recommendation indian_liver_patient_dataset
iranian_churn jm1 kdd_cup_09_appetency kick kickstarter labour_inspection_compliance
lending_club_1m ljubljana_breast_cancer ljubljana_primary_tumor lung_cancer
lung_cancer_epithelial_genexp maps_router_eta_1m marketing_campaign maternal_health_risk
mercari_price_suggestion_1m mercedes_benz_greener_manufacturing miami_housing mic
mice_protein_trisomy_discriminant micro_mass musk mutual_funds_india
naticusdroid_android_permissions_dataset obesity_estimation
online_shoppers_purchasing_intention_dataset otto_group_product_classification_challenge
pancreatic_cancer_mouse_detection parkinsons_biomedical_voice_measurements
physiochemical_protein polish_companies_bankruptcy porto_seguro
predict_students_dropout_and_academic_success prostate_cancer_detection
pva_revenue_prediction_kddcup98 qsar_aquatic_toxicity qsar_biodeg qsar_fish_toxicity
qsar_tid_11 regensburg_pediatric_appendicitis rossmann_store_sales
santander_customer_satisfaction santander_customer_transaction_prediction
santander_transaction_value sat11_hand_algo_runtime sberbank_housing_market_forecasting sdss_17
seismic_bumps sepsis_prediction_1m sepsis_survival_minimal_clinical_records sf_permit_time
south_africa_coronary_heart_disease splice student_portuguese_performance superconductivity
taiwanese_bankruptcy_prediction telemonitoring_parkinsons_biomedical_voice_measurements
thyroid_discordant tour_travels_churn video_game_fps_prediction
video_transcoding_time_prediction website_phishing wids_diabetes_mellitus wine_quality
wine_world_cost
"""
BEYONDARENA: tuple[str, ...] = tuple(_BEYONDARENA_TEXT.split())

#: The OpenML dataset id each SUITE entry resolves to at its pinned version, read from the local
#: scikit-learn OpenML cache (``~/scikit_learn_data/openml``) on 2026-10-16. A new SUITE entry
#: must add its id here, which the first test enforces.
SUITE_OPENML_IDS: dict[str, int] = {
    "mammography": 310, "wilt": 1570, "ozone-level-8hr": 1487, "pc4": 1049, "kc1": 1067,
    "MagicTelescope": 1120, "phoneme": 1489, "spambase": 44, "banknote-authentication": 1462,
}

#: Shortest normalised name for which containment, not only equality, counts as a match:
#: BeyondArena's ``blood_transfusion`` is TabArena's ``blood-transfusion-service-center``, while
#: very short names (``mic``, ``jm1``) would match unrelated datasets by accident.
MIN_CONTAINED = 6


def _norm(name: str) -> str:
    """Lower-case a dataset name and keep only letters and digits, so separators never matter."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _same(a: str, b: str) -> bool:
    """Whether two dataset names plausibly denote one dataset."""
    x, y = _norm(a), _norm(b)
    short, long_ = sorted((x, y), key=len)
    return x == y or (len(short) >= MIN_CONTAINED and short in long_)


def test_the_embedded_lists_are_complete():
    """Guards the constants themselves, so a truncated copy cannot make the overlap test pass."""
    assert len(TABARENA_V0_1) == 51
    assert len({i for i, _ in TABARENA_V0_1}) == 51
    assert len(set(BEYONDARENA)) == 142
    assert set(SUITE_OPENML_IDS) == {name for name, _ in SUITE}


def test_suite_shares_no_openml_id_or_name_with_tabarena_or_beyondarena():
    """Task DR4.1: no SUITE dataset is a TabArena-v0.1 or BeyondArena dataset."""
    tabarena_ids = {i for i, _ in TABARENA_V0_1}
    tabarena_names = [n for _, n in TABARENA_V0_1]
    clashes = []
    for name, _version in SUITE:
        if SUITE_OPENML_IDS.get(name) in tabarena_ids:
            clashes.append((name, "TabArena-v0.1 id"))
        clashes += [(name, f"TabArena-v0.1 {t}") for t in tabarena_names if _same(name, t)]
        clashes += [(name, f"BeyondArena {b}") for b in BEYONDARENA if _same(name, b)]
    assert clashes == []

from fairness.data.objects.Data import Data

class PropublicaRecidivism(Data):

    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'propublica-recidivism'
        self.class_attr = 'two_year_recid'
        self.positive_class_val = 1
        self.sensitive_attrs = ['race'] #, 'sex'] # Joosje: we skip sex
        self.privileged_class_names = ['Caucasian'] #, 'Male']
        self.categorical_features = ['age_cat', 'c_charge_degree', 'c_charge_desc']
        # days_b_screening_arrest, score_text, decile_score, and is_recid will be dropped after
        # data specific processing is done
        self.features_to_keep = ["sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
                                 "juv_other_count", "priors_count", "c_charge_degree",
                                 "c_charge_desc", "decile_score", "score_text", "two_year_recid",
                                 "days_b_screening_arrest", "is_recid"]
        self.missing_val_indicators = []

    def data_specific_processing(self, dataframe):
        # Filter as done here:
        # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        dataframe = dataframe[(dataframe.days_b_screening_arrest <= 30) &
                              (dataframe.days_b_screening_arrest >= -30) &
                              (dataframe.is_recid != -1) &
                              (dataframe.c_charge_degree != '0') &
                              (dataframe.score_text != 'N/A')]
        dataframe['two_year_recid'] = dataframe['two_year_recid'].replace({0: 1, 1: 0}) # Joosje
        dataframe = dataframe.drop(columns = ['days_b_screening_arrest', 'is_recid',
                                              'decile_score', 'score_text'])
        return dataframe

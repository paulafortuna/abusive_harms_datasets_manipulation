

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

generic_path = "/Users/paulafortuna/PycharmProjects/SemEval-2019-Task-6/"

zeerak_folder = "zeerak/"
zeerak_dataset = "dataset_hate_speech_racism_sexism_en_zeerak.csv"
ami_folder = "AMI_misoginy_en_it/"
ami_dataset = "en_training.tsv"
davidson_folder = "davidson/"
davidson_dataset = "dataset_hate_speech_en_davidson.csv"
toxicity_folder = "toxicity/"
toxicity_dataset = "toxicity_en.csv"
stormfront_post_folder = "stormfront_post/"
stormfront_dataset_post = "stormfront_data_annotated_by_post_binary.csv"
trac_train_path = "/home/dtic/SemEval-2019-Task-6/training_short/TRAC_aggressive_en/agr_en_train.csv"
trac_test_path = "/home/dtic/SemEval-2019-Task-6/training_short/TRAC_aggressive_en/agr_en_dev.csv"

pt_fortuna_folder = "multilingual_datasets/pt/"
pt_fortuna_dataset = "hate_speech_pt.csv"
es_ami_folder = "multilingual_datasets/es/"
es_ami_dataset = "es_AMI_TrainingSet_NEW.tsv"
it_ami_folder = "multilingual_datasets/it/"
it_ami_dataset = "it_training.tsv"


def create_dataset(folder, dataset_name, sep):
    reading_path = generic_path + folder + dataset_name
    df = pd.read_csv(reading_path, sep = sep)
    train, test = train_test_split(df, test_size=0.3)
    train.to_csv(generic_path + folder + 'train.tsv', sep = '\t')
    test.to_csv(generic_path + folder + 'test.tsv', sep = '\t')


create_dataset(zeerak_folder, zeerak_dataset, ',')
create_dataset(ami_folder, ami_dataset, '\t')
create_dataset(davidson_folder, davidson_dataset, ',')
create_dataset(toxicity_folder, toxicity_dataset, ',')
create_dataset(stormfront_post_folder, stormfront_dataset_post, ',')

create_dataset(pt_fortuna_folder, pt_fortuna_dataset, ',')
create_dataset(es_ami_folder, es_ami_dataset, '\t')
create_dataset(it_ami_folder, it_ami_dataset, '\t')



import pandas as pd 
import os 
import re
NUM_CLASSES=5
NUM_FOLDS= 10

pattern1 = re.compile(r'S\d+')
pattern2 = re.compile(r'\d+_S_\d+')

def create_labels(df):
    print("Read labels")
    labels=df[['Subject','T1.SERIESID','Group']].copy()

    labels=labels[(labels.Group == 'CN')|(labels.Group == 'AD')|(labels.Group == 'EMCI')|(labels.Group == 'LMCI')|(labels.Group == 'MCI')]

    labels['new_ind']=labels['Subject']+"_"+labels['T1.SERIESID'].astype(str)
    labels.loc[labels.Group == 'CN','Group']=0
    labels.loc[labels.Group == 'AD','Group']=1
    labels.loc[labels.Group == 'EMCI','Group']=2
    labels.loc[labels.Group == 'LMCI','Group']=3
    labels.loc[labels.Group == 'MCI','Group']=4
    labels = labels.set_index('new_ind')
    labels=labels.drop(columns=['Subject','T1.SERIESID'])
    return labels

def create_full_dataset(labels, data_path):
    print("Create full data set: for each MRI, attach a label to it")
    file_names=pd.DataFrame()
    for filename in os.listdir(data_path):
        row=dict()
        row['filename']=filename
        try: 
            series_id = re.search(pattern1, filename).group(0)[1:]
            subject = re.search(pattern2,filename).group(0)
            ind=subject+"_"+series_id
            row['ind']=ind
            row['label']=labels.loc[ind,'Group'] 
        except:
            row['ind']= 'No label'
            row['label']= 'No label'  

        file_names=file_names.append(row,ignore_index=True)

    file_names = file_names[file_names['ind']!='No label']
    print("Done")
    return file_names

def augment_dataset(full_dataset):
    print('Augment the dataset')
    target_count = full_dataset.label.value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    print('Class 2:', target_count[2])
    print('Class 3:', target_count[3])
    print('Class 4:', target_count[4])
    target_count.plot(kind='bar', title='Count (target)')
    
    df_class_0 = full_dataset[full_dataset['label']==0]
    
    #Augment all underrepresented classes
    
    missing_element_1 = target_count[0] - target_count[1]
    df_class_1 = full_dataset[full_dataset['label']==1]
    df_class_1_complementary = df_class_1.sample(missing_element_1, replace=True, random_state=2)
    df_class_1_final = pd.concat([df_class_1, df_class_1_complementary], axis=0)
    
    missing_element_2 = target_count[0] - target_count[2]
    df_class_2 = full_dataset[full_dataset['label']==2]
    df_class_2_complementary = df_class_2.sample(missing_element_2, replace=True, random_state=4)
    df_class_2_final = pd.concat([df_class_2, df_class_2_complementary], axis=0)
    
    missing_element_3 = target_count[0] - target_count[3]
    df_class_3 = full_dataset[full_dataset['label']==3]
    df_class_3_complementary = df_class_3.sample(missing_element_3, replace=True, random_state=8)
    df_class_3_final = pd.concat([df_class_3, df_class_3_complementary], axis=0)
    
    missing_element_4 = target_count[0] - target_count[4]
    df_class_4 = full_dataset[full_dataset['label']==4]
    df_class_4_complementary = df_class_4.sample(missing_element_4, replace=True, random_state=16)
    df_class_4_final = pd.concat([df_class_4, df_class_4_complementary], axis=0)
    
    
    
    dataset_augmented = pd.concat([df_class_0, df_class_1_final, df_class_2_final,
                               df_class_3_final, df_class_4_final], axis=0).sample(frac=1, random_state=64).reset_index(drop=True)
    return dataset_augmented



if __name__ == "__main__":
    df = pd.read_csv("labels.csv")
    labels = create_labels(df)
    full_dataset = create_full_dataset(labels, "./data")
    dataset_augmented=augment_dataset(full_dataset)
    SIZE = len(dataset_augmented)
    #create the 10 datasets:
    slice_size = SIZE//NUM_FOLDS 
    
    for i in range(NUM_FOLDS):
        print("Create the dataset " + str(i))
        data_fold_path = "./datafolds/datafold_"+str(i)
        test_set = dataset_augmented.iloc[slice_size * i : slice_size * (i+1)]
        training_set = pd.concat([dataset_augmented.iloc[slice_size * (i+1):], 
                                 dataset_augmented.iloc[:slice_size * i]], axis=0)
        training_set.to_csv(data_fold_path+"/train_set.csv")
        test_set.to_csv(data_fold_path+"/test_set.csv")
    

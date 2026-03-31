import apply_borderline_smote
import apply_hybrid_sampling
import apply_oversampling
import apply_smote
import apply_smote_enn
import apply_undersampling

if __name__ == '__main__':

    dataset = ["yeast1-5-fold", "yeast3-5-fold", "yeast4-5-fold", "yeast6-5-fold"]
    base_path = r".\dataset"

    for dataset_name in dataset:
        print(f"Processing dataset: {dataset_name}")
        
        apply_smote.main(base_path, dataset_name)
        apply_smote_enn.main(base_path, dataset_name)
        apply_borderline_smote.main(base_path, dataset_name)
        apply_hybrid_sampling.main(base_path, dataset_name)
        apply_oversampling.main(base_path, dataset_name)
        apply_undersampling.main(base_path, dataset_name)

    print("Done")
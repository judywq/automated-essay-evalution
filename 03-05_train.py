from lib.finetuning import FineTuningHelper
import setting

def main():
    helper = FineTuningHelper()
    helper.run(
        training_file_name=setting.dataset_train_filename,
        validation_file_name=setting.dataset_val_filename,
    )

if __name__ == "__main__":
    main()

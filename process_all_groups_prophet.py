import logging
from Exception import CustomException

def process_all_groups_Prophet(df_hol, test_data_hol, all_groups, save_interval=100,
                               save_path='predicted_prophet_data_v1'):
    try:
        all_predicted_data = []
        batch_counter = 0

        for i, group in enumerate(tqdm.tqdm(all_groups, desc="Processing Groups"), start=1):
            try:
                logging.info(f"Processing group {group} (Index {i})...")
                model, future_dfs, predicted_df, train_data_f, test_data = Train_Predict_Prophet(df_hol, test_data_hol,
                                                                                                 group)
                logging.info(f"Successfully processed group {group}")
                all_predicted_data.append(predicted_df)
            except Exception as e:
                raise CustomException(e, sys)
                logging.error(f"Error processing group {group}: {e}")
                continue

            if i % save_interval == 0:
                batch_counter += 1
                try:
                    logging.info(f"Saving batch {batch_counter}...")
                    predicted_data = save_batch_data(all_predicted_data, save_path, batch_counter)
                    all_predicted_data.clear()
                    logging.info(f"Batch {batch_counter} saved successfully")
                except Exception as e:
                    logging.error(f"Error saving batch {batch_counter}: {e}")

        if all_predicted_data:
            batch_counter += 1
            try:
                logging.info(f"Saving final batch {batch_counter}...")
                predicted_data = save_batch_data(all_predicted_data, save_path, batch_counter)
                logging.info(f"Final batch {batch_counter} saved successfully")
            except Exception as e:
                logging.error(f"Error saving final batch {batch_counter}: {e}")

        # Now the return happens after all groups are processed
        return predicted_data
    except Exception as e:
        raise CustomException(e, sys)

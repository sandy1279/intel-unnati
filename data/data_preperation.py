from datasets import load_dataset, Dataset

def prepare_train_data(data_id):
    data = load_dataset(data_id, split="train")
    data_df = data.to_pandas()
    data_df["text"] = data_df[["description", "color"]].apply(lambda x: "\n" + x["description"] + "</s>\n\n" + x["color"] + "</s>", axis=1)
    data = Dataset.from_pandas(data_df)
    return data

if __name__ == "__main__":
    dataset = "burkelibbey/colors"
    data = prepare_train_data(dataset)
    data.save_to_disk("data/color_data")

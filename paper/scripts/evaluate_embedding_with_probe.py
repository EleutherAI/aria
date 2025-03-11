import argparse


from aria.embeddings.evaluate import (
    train_classifier,
    evaluate_classifier,
    CATEGORY_TAGS,
)

EMBEDDING_SIZE = {
    "aria": 512,
    "m3": 768,
    "mert": 1024,
}


def evaluate_embeddings(
    model_name: str,
    metadata_category: str,
    train_dataset_path: str,
    test_dataset_path: str,
    num_epochs: str,
    batch_size: str,
):
    embedding_size = EMBEDDING_SIZE[model_name]
    tag_to_id = CATEGORY_TAGS[metadata_category]

    model = train_classifier(
        embedding_dimension=embedding_size,
        train_dataset_path=train_dataset_path,
        metadata_category=metadata_category,
        tag_to_id=tag_to_id,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )
    evaluate_classifier(
        model=model,
        evaluation_dataset_path=test_dataset_path,
        metadata_category=metadata_category,
        tag_to_id=tag_to_id,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate embeddings with linear prob"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["aria", "mert", "m3"],
        required=True,
    )
    parser.add_argument(
        "--metadata_category",
        type=str,
        choices=["genre", "music_period", "composer", "form", "pianist"],
        required=True,
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch_size for training classifier",
    )
    args = parser.parse_args()

    evaluate_embeddings(
        model_name=args.model,
        metadata_category=args.metadata_category,
        train_dataset_path=args.train_dataset_path,
        test_dataset_path=args.test_dataset_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

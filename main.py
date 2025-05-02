from processor.processor import ProteinDataProcessor
from pipeline.esm_rf_pipeline import ESMRandomForestPipeline
import os

def main():
    processor = ProteinDataProcessor()
    processor.process_all()
    print("Done processing raw files")

    pipeline = ESMRandomForestPipeline()

    # Check if model exists before training
    if not os.path.exists(pipeline.classifier_path):
        print("Training new classifier...")
        pipeline.train(training_set=processor.training_set)
        print("Done training.")
    else:
        print("Loading existing model...")
        pipeline.load_model()

    pipeline.evaluate(validation_set=processor.validation_set)

    # # pipeline.predict(testing_set=processor.testing_set)

if __name__ == "__main__":
    main()

from processor.processor import ProteinDataProcessor as processor
from pipeline.esm_rf_pipeline import ESMRandomForestPipeline as pipeline

def main():
    processor.process_all()
    pipeline.train(training_set=processor.training_set)
    pipeline.evaluate(validation_set=processor.validation_set)
    # pipeline.predict(testing_set)

if __name__ == "__main__":
    main()
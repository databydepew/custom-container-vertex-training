from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
import config

aiplatform.init(project=config.PROJECT_ID, location=config.REGION, staging_bucket=config.BUCKET_URI)

#region vertexai CustomJob
job = aiplatform.CustomJob(
    display_name="test-hptune",
    worker_pool_specs=[
        {
            "machine_spec": {
                "machine_type": config.MACHINE_TYPE,
                #"accelerator_type": accelerator_type,
                #"accelerator_count": accelerator_count,
            },
            "replica_count": config.REPLICA_COUNT,
            "container_spec": {
                "image_uri": "us-central1-docker.pkg.dev/mdepew-assets/ml-repo/supersimple:latest",
                "args": [
                    "python", 
                    "-m", 
                    "trainer.task", 
                    "--dataset_dir", 
                    config.DATASET_DIR,
                    "--hypertune",
                    "True",
                    "--model_dir",
                    config.MODEL_DIR,
                    "--max_depth",
                    "1"
                    ]
            },
        },
    ],
    labels= {
        "ai-flex": "hcustom-train"
        },
    base_output_dir=config.MODEL_DIR
)
#endregion

hp_job = aiplatform.HyperparameterTuningJob(
    display_name='hp-test',
    custom_job=job,
    metric_spec={
        'roc_auc_score': 'maximize',
    },
    parameter_spec={
        'learning_rate': hpt.DoubleParameterSpec(min=0.001, max=0.1, scale='log'),
    },
    max_trial_count=4,
    parallel_trial_count=4,
    labels={'ai-flex': 'hcustom-train'},
    )

hp_job.run()

best = (None, None, None, 0.0)
for trial in hp_job.trials:
    # Keep track of the best outcome
    if float(trial.final_measurement.metrics[0].value) > best[3]:
        try:
            best = (
                trial.id,
                float(trial.parameters[0].value),
                float(trial.parameters[1].value),
                float(trial.final_measurement.metrics[0].value),
            )
        except:
            best = (
                trial.id,
                float(trial.parameters[0].value),
                None,
                float(trial.final_measurement.metrics[0].value),
            )

print(best)
best_mod = best[0]

# Lint as: python2, python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
from typing import Text

from tfx.proto import example_gen_pb2
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components import Transform
from tfx.components import Trainer
from tfx.proto import trainer_pb2
import tensorflow_model_analysis as tfma
from tfx.components import Evaluator
from tfx.components import Pusher
from tfx.proto import pusher_pb2
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from tfx.utils.dsl_utils import tfrecord_input
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen

pipeline_name = 'license_plate'
plate_root = os.path.join(os.environ['HOME'], 'airflow')
data_root = os.path.join(plate_root, 'data', 'tfrecord')
module_file = os.path.join(plate_root, 'dags', 'license_plate_utils.py')
serving_model_dir = os.path.join(plate_root, 'serving_model', pipeline_name)

tfx_root = os.path.join(plate_root, 'tfx')
pipeline_root = os.path.join(tfx_root, 'pipelines', pipeline_name)
metadata_path = os.path.join(tfx_root, 'metadata', pipeline_name, 'metadata.db')

airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}

def create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                    module_file: Text, serving_model_dir: Text,
                    metadata_path: Text,
                    direct_num_workers: int) -> pipeline.Pipeline:
  output = example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(splits=[
          example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=3),
          example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
      ]))
  examples = tfrecord_input(data_root)
  example_gen = ImportExampleGen(input=examples, output_config=output)
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples']) 
  infer_schema = SchemaGen( 
      statistics=statistics_gen.outputs['statistics'], 
      infer_feature_shape=True) 
  validate_stats = ExampleValidator( 
      statistics=statistics_gen.outputs['statistics'], 
      schema=infer_schema.outputs['schema']) 
  transform = Transform( 
      examples=example_gen.outputs['examples'], 
      schema=infer_schema.outputs['schema'], 
      module_file=module_file) 
  trainer = Trainer( 
      module_file=module_file, 
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'], 
      schema=infer_schema.outputs['schema'], 
      train_args=trainer_pb2.TrainArgs(num_steps=100), 
      eval_args=trainer_pb2.EvalArgs(num_steps=50)) 

  eval_config = tfma.EvalConfig(
      slicing_specs=[tfma.SlicingSpec()]
  )

  model_analyzer = Evaluator( 
      examples=example_gen.outputs['examples'], 
      model=trainer.outputs['model'], 
      eval_config=eval_config) 

  model_validator = ModelValidator(examples=example_gen.outputs['examples'],
                                   model=trainer.outputs['model'])
  pusher = Pusher( 
      model=trainer.outputs['model'], 
      model_blessing=model_analyzer.outputs['blessing'], 
      push_destination=pusher_pb2.PushDestination( 
          filesystem=pusher_pb2.PushDestination.Filesystem( 
              base_directory=serving_model_dir))) 

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
          statistics_gen,
          infer_schema,
          validate_stats,
          transform,
          trainer,
          model_analyzer,
          model_validator,
          pusher,
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      beam_pipeline_args=['--direct_num_workers=%d' % direct_num_workers])


DAG = AirflowDagRunner(AirflowPipelineConfig(airflow_config)).run(
    create_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        data_root=data_root,
        module_file=module_file,
        serving_model_dir=serving_model_dir,
        metadata_path=metadata_path,
        direct_num_workers=0))

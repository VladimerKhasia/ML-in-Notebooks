## References and resources for further learning:

- https://www.youtube.com/watch?v=NgWujOrCZFo&list=PLkDaE6sCZn6GMoA0wbpJLi3t34Gd8l0aK

- https://www.deeplearning.ai/courses/machine-learning-engineering-for-production-mlops/

- https://github.com/https-deeplearning-ai/machine-learning-engineering-for-production-public

- https://stanford-cs329s.github.io/syllabus.html

- https://madewithml.com/

- https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022-labs

- https://www.youtube.com/watch?v=LdLFJUlPa4Y&list=PLBoQnSflObckkY7EzV02jifGKgwtBdBW8



## Hands-on Labs:

- deeplearning.ai exercises: https://github.com/https-deeplearning-ai/machine-learning-engineering-for-production-public 

- google cloud:   https://www.cloudskillsboost.google/focuses/2794?parent=catalog

- distributed multi-worker tensorflow training on kubernetes: https://www.cloudskillsboost.google/focuses/17646?parent=catalog

- machine learning with tensorflow in vertex AI: https://www.cloudskillsboost.google/focuses/3391?parent=catalog 

- autoscaling tensorFlow model deployments with TF Serving and Kubernetes: https://www.cloudskillsboost.google/focuses/17649?parent=catalog

- Implementing Canary Releases of TensorFlow Model Deployments with Kubernetes and Anthos Service Mesh:
  https://www.cloudskillsboost.google/focuses/18471?parent=catalog 

- Data Loss Prevention: Qwik Start - JSON: https://www.cloudskillsboost.google/focuses/600?parent=catalog

- check other labs: [here](https://cloud.google.com/compute/docs?utm_source=google&utm_medium=cpc&utm_campaign=emea-none-all-en-dr-sitelink-all-all-trial-p-gcp-1011340&utm_content=text-ad-none-any-DEV_c-CRE_574805387431-ADGP_Hybrid%20%7C%20BKWS%20-%20PHR%20%7C%20Txt%20~%20Compute%20~%20Compute%20Engine%23v1-KWID_43700069710114266-kwd-295775030776-userloc_1007469&utm_term=KW_google%20cloud%20computing-ST_google%20cloud%20computing-NET_g-PLAC_&gclid=CjwKCAiA9NGfBhBvEiwAq5vSy1FFvb2AB4aEfCXkYGN_3i7KRsIUBkpUEXmI2MkgnOoYbvkF7ftnshoC5YMQAvD_BwE&gclsrc=aw.ds)



## Handouts

**PART 1 - General:**

concept and data drift 

houseprice example: inflation caused houseprice change (concept) vs only large houses are built (data)

(there is also covariate shift/notdrift when x and y distributions change but x-->y mapping does not)

- track software-->memory,compute,latency,throughput, server load
- track data (input/output) matrics e.g. withdashboards-->input:num missing values;output:non-nulls etc.

image labeling tool - https://landing.ai/platform/

ML experiment tracking tools: weights and biases, comet ml, mlflow, sagemaker studio, landing.ai etc.

Good Data: is big enough; covers important inputs; is labeled consistently; concept/data drift covered.

<br>

**PART 2 - Data Handling:**

General and Data Collection and papers:

- https://cd.foundation/blog/2020/02/11/announcing-the-cd-foundation-mlops-sig/

- https://karpathy.medium.com/software-2-0-a64152b37c35

- https://pair.withgoogle.com/chapter/data-collection/

- https://developers.google.com/machine-learning/guides/rules-of-ml

- https://ai.googleblog.com/2018/09/introducing-inclusive-images-competition.html

- https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf

- https://arxiv.org/abs/2010.02013

process feedback labling:

(1.) automatic labling of live data - apply label based on prediction

(2.) label can be created based on some logs we deigned with logging tools, like:

- https://www.elastic.co/logstash/ 

- https://www.fluentd.org/

- for cloud:  https://cloud.google.com/logging, aws elasticsearch, azure monitor


Tensorflow: exampleGen splits data --> statisticGen produces mean std etc.--> schemaGen checks data types

<br>

TFDV - tensorflow data validation:

(measures skews is measured in chebishev distance max(|a-b|) in every single dimension)

1. schema skew        - change in data types int vs float etc. 

2. feature skew       - when feature values change

3. distribution skew  - inside individual feature distribution changes (mean, std etc.)

	https://blog.tensorflow.org/2018/09/introducing-tensorflow-data-validation.html

	https://en.wikipedia.org/wiki/Chebyshev_distance	


Tensorflow Transform For Feature Engineering:

does feature engineering. tf.transform Analyzers do that.

e.g. tft.scale_to_z_score(...) for standartizing.

tf_metadata for storing data types and more

<br>

Feature Engineering:

   squeeze most information out of data with least features 

	scaling((a-b)/b), if not gaussian normalizing 0 to 1 ((a-min)/(max-min)),standartizing(z-score)

	grouping: bucketizing, bag of words

	dimensionality reduction: pca, t-sne, uniform manifold aproximation and projection (umap), 

	custom linear projections. Tool for getting feature insights https://projector.tensorflow.org/

	feature crossing: devide, multiply many etc. If categorical combine week+time

<br>

Feature Selection:

unsupervised casses --> e.g. discard or make one feature from two/more correlated features

supervised casses --> idea is to select those that cause right predictions the most!

   There are three method groups>>

   1. Filter Methods: 

	- correlation> 

	   pearson correlation: linear relationships, feature-feature corr is bad and remove,feature-target corr is good.

	   kendall tau rank correlation coefficient: monotonic relationships + small sample.

	   spearman's rank correlation coefficient: monotonic relationships

	   mutual information

	   F-test

	   Chi-squared test

	- univariate feature selection> 

	   means that each features correlation with target is assessed separately (Anova)

	   Sklearn commands: 1. SelectKBest, 2. SelectPercentile, 3. GenericUnivariateSelect

           Sklearn assisting test commands:

		for regression-->     mutual_info_regression, f_regression

		for classification--> mutual_info_classif, f_classif, chi2

   2. Wrapper Methods:

        forward elimination            --> selects good features one-by-one - from zero up

        backward elimination           --> removes least important features - from all down

        recursive feature elimination  --> we define how many features we want and recursively eliminate least important ones

   3. Embeded Methods: 

        L1 Regularization

        Feature Importance


Feature engineering and selection references:

https://developers.google.com/machine-learning/crash-course/representation/feature-engineering

https://pair-code.github.io/facets/

http://projector.tensorflow.org/

https://developers.google.com/machine-learning/crash-course/feature-crosses/encoding-nonlinearity

TFX references:

https://www.tensorflow.org/tfx/guide#tfx_pipelines

https://ai.googleblog.com/2017/02/preprocessing-for-machine-learning-with.html


<br>

Data Leneage:

ML Metadata from TFX to manage whole data life: 

https://www.tensorflow.org/tfx/guide/mlmd

https://www.tensorflow.org/tfx/guide/understanding_custom_components

   - metadata component:

	driver   >> gets the required metadata

	executor >> applies functionality based of provided metadata (provided by driver)

	publisher>> stores new result into metadata 

   - artefacts(data,model..), executions(action), contexst(what execution on which artifact)	

   - list, graph and compare artifacts(current or old)

other data versioning tools:   https://dvc.org/ , https://git-lfs.com/

Schema includes: feature names, types, required vs optional, one vs multyple value in feature etc.

Features: precomputed time to time and stored, used for inference.

- https://cloud.google.com/blog/products/ai-machine-learning/introducing-feast-an-open-source-feature-store-for-machine-learning

- https://github.com/feast-dev/feast

- https://www.gojek.io/blog/feast-bridging-ml-models-and-data

Data warehouse: 

differs from database because: 1. It's for analytical purpose 2.also stores historical data 

differs from data lake, which is just raw data storage

<br>

Advanced Labeling:

1. semi-supervised learning:

   label small part of data + 

   label propagation to the (in the feature space) nearby unlabeled data 

   (propagate labels of your labeled data to other unlabeled data based on

   how near is an unlabeled datapoint to labeled one in the feature space)

   https://arxiv.org/pdf/1904.04717.pdf

2. active learning:

   label and learn what you do not know by >>

   -margine sampling  --> label points that are near decission boundry, so are less certain

   -querry-by-cometee --> human labling where dissagreement or low certainty occurs

   -cluster-based sampling--> "elongates" clusters to see where their actual boundaries are

   -region-based sampling --> devides space into many regions and applies different active learning algorithms

3. weak supervision

   use several labeling logics (e.g. if contains some words is spam) at once (labling functions)

   combine those labling function results L into single probabilistic result, label probabilities

   this is done by neural network(refered as generator), which 

   learns overlaps and disagreements of input L's  in just one pass forward+backward,loss is 

   calculated via negative log likelihood. So where signals combine/agree is left almost

   unchanged and where signals disagree that outcomes get even more penalized and so noisy.
   
   apply those label probabilities like actual labels for your model (called discriminator)

   (chop off last layer of your model and make output corresponding to that 

   label probabilities if needed)

   https://www.youtube.com/watch?v=SS9fvo8mR9Y&list=LL&index=2

   https://www.youtube.com/watch?v=M7r5SGIxxpI&list=LL&index=2

   Paper: "Weak supervision: the new programming paradigm for machine learning" Alex Ratner et al

      https://ajratner.github.io/assets/papers/deem-metal-prototype.pdf

      https://ajratner.github.io/ 

   actual tool is snorkel: 

	https://www.snorkel.org/

	https://snorkel.ai/weak-supervision/#:~:text=Weak%20supervision%20is%20an%20approach,manually%2C%20one%20by%20one).

   just augment already labeled data

   - unsupervised data augmentation: https://github.com/google-research/uda

   - auto augmentation: https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html

   method removing useless data: https://www.deeplearning.ai/the-batch/new-method-removes-useless-machine-learning-data/?utm_campaign=The%20Batch&utm_content=238362150&utm_medium=social&utm_source=twitter&hss_channel=tw-992153930095251456

<br>

**PART 3 - Model Handling:**


AutoML:

   1. Neural Architecture Search Strategies >>

        https://arxiv.org/pdf/1808.05377.pdf

        https://distill.pub/2020/bayesian-optimization/

        https://arxiv.org/pdf/1611.01578.pdf

        https://arxiv.org/pdf/1712.00559.pdf

        https://arxiv.org/abs/1603.01670

   2. Tool e.g. keras-tuner

   3. Tools on cloud:

        amazon sagemaker autopilot: https://aws.amazon.com/sagemaker/autopilot/?sagemaker-data-wrangler-whats-new.sort-by=item.additionalFields.postDateTime&sagemaker-data-wrangler-whats-new.sort-order=desc

        microsoft azure automated machine learning:  https://azure.microsoft.com/en-in/products/machine-learning/automatedml/ 

        google cloud automl: https://cloud.google.com/automl

   4. some hands-on labs: 

		https://www.cloudskillsboost.google/focuses/2794?parent=catalog

		[find more here](https://cloud.google.com/compute/docs?utm_source=google&utm_medium=cpc&utm_campaign=emea-none-all-en-dr-sitelink-all-all-trial-p-gcp-1011340&utm_content=text-ad-none-any-DEV_c-CRE_574805387431-ADGP_Hybrid%20%7C%20BKWS%20-%20PHR%20%7C%20Txt%20~%20Compute%20~%20Compute%20Engine%23v1-KWID_43700069710114266-kwd-295775030776-userloc_1007469&utm_term=KW_google%20cloud%20computing-ST_google%20cloud%20computing-NET_g-PLAC_&gclid=CjwKCAiA9NGfBhBvEiwAq5vSy1FFvb2AB4aEfCXkYGN_3i7KRsIUBkpUEXmI2MkgnOoYbvkF7ftnshoC5YMQAvD_BwE&gclsrc=aw.ds)

<br>

Lowering Resource Requirements:

   1. projection subspace examples >>

        linear discriminant analysis LDA - classification case

        partial least squares PLS - regression case

        principal component analysis PCA - unsupervised case variance

        https://arxiv.org/pdf/1404.1100.pdf

        http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/

        other unsupervized dimnsionaity reduction technics:

        latent semantic indexing/analysis LSI/LSA - singular value decomposition SVD

        unlike PCA the SVD can be used for nonsquare and sparce matrices 

        independent component analysis ICA

        https://arxiv.org/pdf/1404.2986.pdf

        if PCA removes correlations ICA removes correlations and higher order dependence

        and for ICA all components are equally important unlike PCA

        matrix factorization: non-negative matrix factorization NMF
        
        unlike PCA NMF is interpretable but operates only on non-negative data

        latent methods: latent dirichlet allocation LDA

   2. quantization >>

        about quantization:          https://arxiv.org/abs/1712.05877

        post training quantisation:  https://medium.com/tensorflow/introducing-the-model-optimization-toolkit-for-tensorflow-254aca1ba0a3

        quantization aware training: https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html

   3. pruning >>

        http://yann.lecun.com/exdb/publis/pdf/lecun-90b.pdf

        https://arxiv.org/abs/1803.03635


<br>

Distributed Training:

   1. data parallelism >>

        synchronous training: all-reduce architecture

        asynchronous training: parameter server atchitecture(slower convergence,low accuracy)

        tolls: e.g. tensorflow.distribute.Strategy

        https://www.tensorflow.org/guide/distributed_training

        strategies: e.g. 

            one device        - just for testing basically, no distribution

            mirrored          - one machine multiple gpu-s

            parameter server  - multiple machines (so asinchronous by nature)

            etc . . .

            fault tollerance: if one worker fails others wait until it restarts

   2. model parallelism >>

        grouped model layers in similar model complexity chunks are distributed across workers

        downside is sequential nature of training.

   3. pipeline parralelism >> 

        uses all togehted data and model paralelism and more +

        gradient accumulation: from mini to micro batches-accumulated gradients are used for backprop

        tools: 

        PipeDream -- https://arxiv.org/abs/1806.03377, https://www.microsoft.com/en-us/research/blog/pipedream-a-more-effective-way-to-train-deep-neural-networks-using-pipeline-parallelism/

        GPipe     -- https://arxiv.org/abs/1811.06965 , https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html

   4. Hands-on exercise: https://www.cloudskillsboost.google/focuses/17646?parent=catalog

<br>

Knowledge Distillation:

   1. for downsizing model with no/least performance loss:

      - smaler student model uses bigger teachers soft logits as targets (mostly KL divergence)
        and actual hard labels as well to calculate its own loss. both parts of loss
        (divergence from teacher and from actual labels) are weighet by 1-a and a
        
        https://arxiv.org/pdf/1503.02531.pdf

        [for myself: for the intuition why it works remember proximal policy optimization but not because of KL]

      - we can combine: pretraining with multytask teacher, then training by multiple different 
        architecture teachers, then assambly all students together. (two-stage distillation)

	     https://arxiv.org/pdf/1910.08381.pdf 
        
   2. putting more knowledge into teacher using bigger noisy/moregeneralizable student in loop

	   https://arxiv.org/abs/1911.04252

      [for myself: intuition is larger noisy student acts like exploration in RL,
      it learns something more and becomes the teacher to pass acquired knowledge.
      starting teacher "explotation" becomes the student adopting the knowledge like value iteration]

<br>

Tensorflow Model Analysis (TFMA):

   resonates with error analysis, because beside overall model evaluation it allows 
   us to inspect the performance on the parts of data. (TFMA uses apache beam)
   TFMA consists of >>

   1. read inputs   --> converts any format to dictionary with dtype is tfma.extracts
   2. extraxtors --> input,slice key,custom and predict extractors do prediction on slices   
   3. evaluators --> MetricsandPlotsEvaluator, AnalysisEvaluator, CustomEvaluator
   4. write results --> just writes results to disk

   - in case of using outside TFX create input config file via tfma.EvalConfig(...) manually

   - https://blog.tensorflow.org/2018/03/introducing-tensorflow-model-analysis.html

   - https://www.tensorflow.org/tfx/model_analysis/architecture

   <br>

   Model Debugging:

   - ojectives: model decay(time,distributionShift), biases(e.g.discrimination), privacy, security

   - technics: benchmarking, sensitivity analysis, residual analysis

   1. benchmarking: where does your model perform worse htan simple benchmark model and why

   2. sensitivity analysis: 

      *Main Idea* of sensitivity is how the perturbation/change in feature(s) changes model prediction

      *identify* --> attacs that our model robustnes is sensitive to, can be affected by >>

         - general tool for sensitivity identification: https://pair-code.github.io/what-if-tool/

         - methods for identifying sensitivity against attacs:

            a. random attacks - just provide the model with high volume of random input data

            b. partial dependence plots - visualizes marginal effect of one/more features on model results

            tools for this: https://github.com/SauceCat/PDPbox and https://github.com/AustinRochford/PyCEbox

            c. adversarial attacs - carefuly designed changes/distortion (e.g. in features) https://arxiv.org/abs/1412.6572

               can cause informational and behavioral harms:

               *informational harms*: recreate persons data, whole model data, or model itself

               correspondingly called membership inference, model inversion, model extraction.

               *behavioral harms*: evasion(causes missclasification), poisoning(more general term)

            tool for measuring/benchmarking sensitivity: 
            https://github.com/cleverhans-lab/cleverhans

      *remediate the model* --> 

         - by adversarial training with designed adversarial attack data.

         - tool and resource: https://github.com/bethgelab/foolbox

         - one good method: deffnsive distillation

            like knowledge distillation but same model architecture is used 

            for both student and teacher https://arxiv.org/abs/1511.04508
	
   3. residual analysis: 

      residual in regression - difference between prediction and ground truth

      residual distribution should be random

      if residual distribution correlates with feature that was not used in the feature vector that feature should be included

      if adjacent residuals selfcorelate - one predicts next(just sign correlations also matter)

      *Durbin-Watson test* is often used for detecting the autocorrelation.

 
   4. general model remediation: 

      on data and model level

      data augmentation via generative technics, interpretable technics(data,model insights) or just noise addition.

      model editing(when model understanding enables succesfull manual tweeks), model assertions(age is allways positive)

      tools: https://www.tensorflow.org/responsible_ai/model_remediation 

   5. fairness:  tools and resources

      https://arxiv.org/pdf/1904.13341.pdf

      https://www.tensorflow.org/responsible_ai/fairness_indicators/guide

      https://www.tensorflow.org/responsible_ai/model_remediation

      https://github.com/cosmicBboy/themis-ml

      https://modelcards.withgoogle.com/about

      http://aif360.mybluemix.net/

 
<br>

Continuous Evaluation and Monitoring:

   1. problem is concept drift, data drift/shift, covariate shift

   2. where can those problems be caught and so what should we monitor:

      - covariate shift is detectable by comparing old and new data itself

      - data drift/shift(prior probability shift) by comparing old and new predictions

      - concept drift only after labling new sample of input data, which is somewhat deleyed process

   3. what are supervised and unsupervised methods of monitoring:

      - supervised: statistical process control, sequencial analysis, error distribution monitoring

      - unsupervised: 

         clustering/novelty detection (algorithms: OLINDDA, MINAS, ECSMiner, GC3)

         feature distribution monitoring

         model-dependent monitoring (algorithms MD3 etc.)

   4. as via monitoring we catch the problem early we retrain the model before problem affects us

   5. tools and resources:

      https://arxiv.org/pdf/1704.00023.pdf

      https://www.infoq.com/presentations/instrumentation-observability-monitoring-ml/

      https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/

      google primarely focuses on concept drift: https://cloud.google.com/ai-platform/prediction/docs/continuous-evaluation

      amazon focuses on concept drift: https://aws.amazon.com/sagemaker/model-monitor/

      microsoft focuses on data drift: https://learn.microsoft.com/en-us/azure/machine-learning/v1/how-to-monitor-datasets?tabs=python



<br>

Model Interpretability:

   1. generally on interpretability:

      method categorization: intrinsic vs post-hoc, model specific vs agnostic, local vs global etc.

      designed to be or by nature intrinsicly interpretable. (mostly by nature: 

      linear regression,logistic regression, tree based, tensorflow Lattice models [uses vertices,linear interpolations], knn, RuleFit)    

      human intuition holds when: feature behaves monotonicly and/or its behavior matches domain knowledge

      resources and tools: 

      https://arxiv.org/pdf/1910.10045.pdf

      https://jmlr.org/papers/volume17/15-243/15-243.pdf

      https://www.tensorflow.org/lattice

      https://christophm.github.io/interpretable-ml-book/ 

   2. some model agnostic methods (mostly overlap with non-intrinsic, post-hoc methods):

      partial dependence plots, augmented local effects, individual conditional expectation,

      permutatin feature importance, local surrogate(LIME) and global surrogate, shap and shapley values, concept activation vectors

      lime - https://github.com/marcotcr/lime

      testing with concept activation vectors - https://arxiv.org/pdf/1711.11279.pdf   

      more undestanding of explanations: 
      
      https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf

      https://arxiv.org/abs/1905.04610

      https://arxiv.org/abs/1801.01489       

      https://storage.googleapis.com/cloud-ai-whitepapers/AI%20Explainability%20Whitepaper.pdf



<br>

**PART 4 - Serving and Maintainance:**

1. Model Serving Infrastructure:

	NoSQL:

	google cloud memory store/amazon dynamoDB - fast lookup,in memory cache
	google cloud firestore - handels slowly changing data
	google cloud bigtable - handels dinamically changing data

	Server: 

	any managed cloud platform, or something like tensorflow serving (gRPC,REST), clipper https://github.com/simon-mo/clipper/tree/develop

	tf.serving two options: tensorflowmodel-server, tensorflow-model-server-universal (includes only basic optimizations, runs on most machines)

	Mobile: average app 11mb, whole gpu 4gb, tweak threads?

2. Scaling:

	scale horizontally, 

	use containers(docker), use container orcestrarion tools (kubernetes, docker swarm)

	use Kubeflow for deploying ML workflows(data ingestion, feature extraction, model management etc) on Kubernetes

3. Extract Transform Load:

	apache beam (+Cloud DataFlow), parts of TFX, apache spark, kafka can be used

	for common casses radymade preprocessed versions of data can be used in cache (e.g. common words etc.)

4. Management:
	- Experiment and Track Experiments: 

      track runtime parameters - in config files or via command line

      convert from .ipynb to .py - using bnconvert, nbdime, jupytext, neptune notebooks (and organize code instead of notebook format)

      use data versioning tools: Neptune, Pachyderm, Delta Lake, Git LFS, Dolt, lakeFS, DVC, ML-Metadata

      use model versioning (major.minor.patch) and model registries(central repository of all model versions):

            1. major for incompatible API changes, minor for backwards compatible new functionalities, patch/pipeline just backwards compatible bug fixes
         
            2. some model repositories: Azure ml model registry, sas model manager, MLflow model registry, google AI platform, algorithmia

      log, tag, organize, make experiments searchable and shareable(neptune ai tools, tensorboard etc.)

      set baseline and get first comparison against it

      define next steps for improvement and for meeting your business goals (latency, cost etc.)

      https://towardsdatascience.com/machine-learning-experiment-tracking-93b796e501b0 and https://neptune.ai/blog/experiment-management

	- Continuous Integration - testing and validation of data, schemes, models etc.

	- Continuous Delivery - automatized deployments, infrastructure validation e.g. with TFX InfraValidator that lunches sandbox-model for 
     validation (e.g. new retrained version etc.)

	- Continuous Monitoring - for catching errors and potential problems early (e.g. catch concept drift, model decay, model security etc.)

	- Continuous Training - automatic retraining and testing of models (e.g. as response to data or concept drift)

	- Progressive Delivery:

		- blue-green deployment (new green version for everyone, older blue as backup), canary deployment (same as blue-green, but new green for 
        only small group and than increase gradually)	

		- live testing with A/B testing (test 2 or multiple versions on 2 or multiple groups), A/B testing with multy armed bandits, or with      
        contextual bandits (e.g. selling clothing in hot vs cold country)

	- more automation is sign of more maturity

   - system performance monitoring:  as you deal with complex system, it is difficult to know which part(s) actualy cause e.g. latency.
     for tracing individual component performances there are so called Dapper style tracing implementation using Dapper, Zipkin, Jaeger. 
     https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/     



0. PART 4 Resources:

   model severs:

   https://www.tensorflow.org/tfx/serving/architecture

   https://github.com/pytorch/serve

   https://www.kubeflow.org/docs/external-add-ons/serving/

   https://developer.nvidia.com/nvidia-triton-inference-server

   <br>
   
   kubernetes and kubeflow:

   https://kubernetes.io/docs/tutorials/kubernetes-basics/

   https://www.youtube.com/watch?v=H06qrNmGqyE

   https://www.kubeflow.org/docs/

   https://www.youtube.com/watch?v=dC659IsHNyg&list=PLIivdWyY5sqLS4lN75RPDEyBgTro_YX7x&index=4

   about scaling: https://www.youtube.com/watch?v=aIxNm5Eed_8

   <br>

   data preprocessing apache beam:

   there is colab version as well: https://beam.apache.org/get-started/try-apache-beam/

   https://beam.apache.org/get-started/wordcount-example/

   https://beam.apache.org/documentation/programming-guide/

   tensorflow using beam style pipeline: https://www.tensorflow.org/tfx/transform/get_started

   <br>

   Autoscaling TensorFlow Model Deployments with TF Serving and Kubernetes:
   https://www.cloudskillsboost.google/focuses/17649?parent=catalog

   <br>

   ML experiment tracking and experiment management:

   https://towardsdatascience.com/machine-learning-experiment-tracking-93b796e501b0

   https://neptune.ai/blog/experiment-management

   <br>

   MLOps methodologies and resources:

   https://neptune.ai/blog/mlops

   https://github.com/visenger/awesome-mlops

   <br>

   MLOps with TFX:
   https://cloud.google.com/architecture/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build


   Model Management:
   https://neptune.ai/blog/machine-learning-model-management


   Continuous Delivery:
   https://continuousdelivery.com/


   Progressive Delivery:
   https://codefresh.io/docs/docs/ci-cd-guides/progressive-delivery/


   Implementing Canary Releases of TensorFlow Model Deployments with Kubernetes and Anthos Service Mesh:
   https://www.cloudskillsboost.google/focuses/18471?parent=catalog 


   Monitoring and Logging:
   https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/


   Data Loss Prevention: Qwik Start - JSON:
   https://www.cloudskillsboost.google/focuses/600?parent=catalog


   Address Model Decay:
   https://neptune.ai/blog/retraining-model-during-deployment-continuous-training-continuous-testing


   Responsible AI:
   https://ai.google/responsibilities/responsible-ai-practices/


   Legal Requirements: 
   GDPR and CCPA e.g. --> https://gdpr.eu/ and https://oag.ca.gov/privacy/ccpa etc.

   <br>

   All Tools and Resources:

   https://cloud.google.com/memorystore

   https://cloud.google.com/firestore

   https://cloud.google.com/bigtable

   https://aws.amazon.com/dynamodb/

   https://arxiv.org/abs/1704.04861

   https://rise.cs.berkeley.edu/projects/clipper/

   https://www.tensorflow.org/tfx/guide/serving

   https://medium.com/@vikati/the-rise-of-the-model-servers-9395522b6c58

   https://www.tensorflow.org/tfx/serving/architecture

   https://developer.nvidia.com/nvidia-triton-inference-server

   https://github.com/pytorch/serve

   https://www.kubeflow.org/docs/external-add-ons/serving/

   https://phoenixnap.com/blog/what-is-container-orchestration

   https://kubernetes.io/

   https://docs.docker.com/engine/swarm/

   https://www.kubeflow.org/

   https://mlinproduction.com/batch-inference-vs-online-inference/

   https://github.com/ertis-research/kafka-ml

   https://cloud.google.com/pubsub

   https://cloud.google.com/dataflow

   https://spark.apache.org/

   https://towardsdatascience.com/machine-learning-experiment-tracking-93b796e501b0

   https://neptune.ai/blog/experiment-management

   https://nbconvert.readthedocs.io/en/latest/

   https://nbdime.readthedocs.io/en/latest/

   https://jupytext.readthedocs.io/en/latest/install.html

   https://docs.neptune.ai/

   https://docs.neptune.ai/tutorials/data_versioning/

   https://www.pachyderm.com/

   https://delta.io/

   https://git-lfs.com/

   https://github.com/dolthub/dolt

   https://lakefs.io/blog/data-versioning/

   https://dvc.org/

   https://blog.tensorflow.org/2021/01/ml-metadata-version-control-for-ml.html

   https://www.tensorflow.org/tensorboard/image_summaries

   https://neptune.ai/pricing

   https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-overview

   https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

   https://blog.tensorflow.org/2020/01/creating-custom-tfx-component.html

   https://github.com/tensorflow/tfx/blob/master/docs/guide/custom_component.md

   https://www.split.io/glossary/progressive-delivery/

   https://launchdarkly.com/blog/continuous-incrementalprogressive-delivery-pick-three/

   https://dev.to/mostlyjason/intro-to-deployment-strategies-blue-green-canary-and-more-3a3

   https://martinfowler.com/bliki/BlueGreenDeployment.html

   https://medium.com/capital-one-tech/the-role-of-a-b-testing-in-the-machine-learning-future-3d2ba035daeb

   Monitoring Related Tools And Resources:

   https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf

   https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/

   https://cloud.google.com/monitoring

   https://aws.amazon.com/cloudwatch/

   https://learn.microsoft.com/en-us/azure/azure-monitor/overview

   https://storage.googleapis.com/pub-tools-public-publication-data/pdf/36356.pdf

   https://www.jaegertracing.io/

   https://zipkin.io/

   https://cloud.google.com/vertex-ai

   https://cloud.google.com/vertex-ai/docs/datasets/label-using-console

   https://www.kdnuggets.com/2020/08/anonymous-anonymized-data.html

   https://dataprivacymanager.net/pseudonymization-according-to-the-gdpr/ 
   
   You might also want to check other somewhat related repositories:
   
   https://github.com/The-AI-Summer/Deep-Learning-In-Production

   Other Google managed Labs and Materials:
   - All: https://www.cloudskillsboost.google/paths
   - ML Engineering: https://www.cloudskillsboost.google/paths/17
   - Generative AI: https://www.cloudskillsboost.google/paths/183 
   - Duet AI & Summit AI: https://www.cloudskillsboost.google/paths/236  and  https://www.cloudskillsboost.google/paths/280

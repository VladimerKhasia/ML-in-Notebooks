## References and resources for further learning:

- https://www.youtube.com/watch?v=NgWujOrCZFo&list=PLkDaE6sCZn6GMoA0wbpJLi3t34Gd8l0aK

- https://www.deeplearning.ai/courses/machine-learning-engineering-for-production-mlops/

- https://stanford-cs329s.github.io/syllabus.html

- https://madewithml.com/

- https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022-labs

- https://www.youtube.com/watch?v=LdLFJUlPa4Y&list=PLBoQnSflObckkY7EzV02jifGKgwtBdBW8



## Hands-on Labs:

- google cloud:   https://www.cloudskillsboost.google/focuses/2794?parent=catalog

- distributed multi-worker tensorflow training on kubernetes: https://www.cloudskillsboost.google/focuses/17646?parent=catalog

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

?'$	(?5'@?綯X?@"rl=?#@!$?F??4@$	?tr?wh"@	?&?O?@D?qP?@!i~LX?x4@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??'I?d&@?/EH?@A?!??7@Y?-Θ??r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?-????$@?Qew?@A??t@Y???V	??r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??????#@-_????@AZ_@Y?[u?)??r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?x?$@KU?2@AF??j?x@Y?%Tpx??r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?46L$@?N??DZ@AO?6??&@Yjܛ?0??r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??????&@??q?j?@A??̰!@Y?#K&??r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	$?F??4@[?:???@A?w??11@Yg?????r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
?????&@ C?J@A/ޏ?/@Y??f?8??r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1\Ɏ?@?'@n5????@A?4??!@Y?t<f????r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1????Ǘ&@E|V??AU/??d#@Y?T????r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???y?X(@??'?b?@A????@YgE?D??@r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?vۅ??'@?????@A?}???@Y$??????r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?]???'@k??=]?@A???4?Q@Y"O??????r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???]%@6?ڋh?@ATƿ??@Y?J??q(??r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??]i&@???????A????'?!@Y?Q<????r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1????x&@??:ǀ?@Au?yƾD@Y{????Z??r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1"rl=?#@CV??@A?nض(?@Y?鷯??r	train 517*	L7?A???@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatYM?]W@!l?2cA@)gE?D?o@1^ق?L?>@:Preprocessing2T
Iterator::Root::ParallelMapV27QKs+??!??YX0$3@)7QKs+??1??YX0$3@:Preprocessing2E
Iterator::Rootk?=&R?@!>??IKB@)??/?xp??1??';f1@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceV*?????!?S??}?0@)V*?????1?S??}?0@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??B?? @!?????O@)?uX???1Ek??\?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(?N>???!??iG??4@)H?V
??1????@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*$0??{??!?Kd}?K@)$0??{??1?Kd}?K@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?U?6????!??#ދ7@)7n1?74??1@?е?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t20.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?>0??I"@I*?9?ǶV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???@?l??E|V??!k??=]?@	!       "	!       *	!       2$	?k?UP @??Wh@?nض(?@!?w??11@:	!       B	!       J$	??3???????M?????[u?)??!gE?D??@R	!       Z$	??3???????M?????[u?)??!gE?D??@b	!       JCPU_ONLYY?>0??I"@b q*?9?ǶV@Y      Y@q??!5J???"?	
both?Your program is MODERATELY input-bound because 9.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t20.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.2no:
Refer to the TF2 Profiler FAQ2"CPU: B 
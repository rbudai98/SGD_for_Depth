$	3)?c?L(@?s???2@֫?耔#@!(}!??S1@$	?d҄?$@?_?W?U@?6???@!E??Fe?A@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1@?3iS?,@~r 
?@AW#??%@Y<hv?[???r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1a??M?*@???h?S@A?_ ?1"@Y£?#V??r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?F? ?%@n?HJz?@A?\?@Yѯ?????r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??0X:(@}?F?&@A?????!@Y.Ȗ?k??r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?????%*@t\??Jk@A@7n?#@YDN_??,??r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1M??.?%@
??:??AU????,"@Ym򖫟??r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	?eN??$@?}8H??@A?~?~?4@Y?l ]lZ??r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1

J?ʽ $@?????@A+n?b~?@Y?D.8????r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???\$@%<?ן?@A??f?R`@YM?d??7??r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1n?r?$@??ݒ?@AŒr?9?@Y??????r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1֫?耔#@X????W@A??c!:?@YJ)??????r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1j/????&@???JY?@A]m???? @Yt]???T??r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1? Ϡ?.@?w??@A???B?y@Y???L??@r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??y??&@ Tq?s??ATn???&!@Y??*4???r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1(}!??S1@?릔?j@Aj2?m??(@Y	4??y?@r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1\*??3*@ʉv?@AOGɫ @Y?}"?@r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?&?+??*@ߥ?%??@A	R)v4?!@Y?Z?a/??r	train 517*	?Q??5?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat5?l?/?@!#c????@@)??AB??@1
"??>@:Preprocessing2T
Iterator::Root::ParallelMapV20???"???!$5?e<G6@)0???"???1$5?e<G6@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???????!?]a?/@)???????1?]a?/@:Preprocessing2E
Iterator::Rootvk??3@!??6?,yA@)??>rk???1`Oq9V)@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip????t?@!??d?iCP@)?릔?J??1^"4X?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???S????!_[!x5@)wg????1?????@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapKi?, @!?1?[8@)??h?????1h?o??
@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*+?WY???!?!??8@)+?WY???1?!??8@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 10.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t20.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?^D???%@I/tw?@DV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	&??ڴ@K^`?j??
??:??!ߥ?%??@	!       "	!       *	!       2$	?? Tv? @Aʵ??.??Œr?9?@!j2?m??(@:	!       B	!       J$	?0??BA??E>L? ???DN_??,??!???L??@R	!       Z$	?0??BA??E>L? ???DN_??,??!???L??@b	!       JCPU_ONLYY?^D???%@b q/tw?@DV@
$	;????L!@Q	???n??	??Y?@!q:?V#@$	??_/t@???9?v	@?ⷓ*???!?:??*@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1????<<!@Ii6?????A5&?\r@Y*?Z^????r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?jGq?j!@?Y?X??Ae ??Ɲ@Y???'??r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??:?z"@]?????@A??-Ii@Y??U????r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1q:?V#@??P????A?>d@Y???????r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1mr???!@?r?}????A???Z8@Y9????l??r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?b??"@r??rg???AuWv???@Y>?^?????r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	?	M?!@??W?2???A?%?"@Yi??????r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
?x? @???o??Ag??V@YF%u???r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1c	kc?|!@??_ ??A?m?@Y??|?r???r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??cꮼ@#/kb???A%???d@YOt]?????r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	??Y?@M??E???A?ܚt[R@Y??1>?^??r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1Z?1?	?!@?6 ???A霟?8?@Y??q??r??r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1հ??!@??1=aI??AN?0???@Y?]K?=??r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?e3? @?#G:???A??ꫫ?@Y?d9	?/??r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	4??y? @??S???A?	h"l?@Y6[y??$??r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?(????!@???????A-z???@Ym??????r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?l??)-!@6<?R???A??tx#@Y?4*p????r	train 517*	?&1e?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??Χ@!?+%\A@)m?Yg|? @1?;?/?@@:Preprocessing2T
Iterator::Root::ParallelMapV2?խ????!?4??sk1@)?խ????1?4??sk1@:Preprocessing2E
Iterator::Rootj1x??? @!??dB?'@@)??$????1??q?q?-@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?º?????!*?$?)@)?º?????1*?$?)@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipk??*?@!.????P@)Թ??,??1?f??M? @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateR?h??!?-E?654@)??ʾ+???1?a???T@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?ފ????!"?	'??8@)fJ?o	???1s???@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?wak????!????#@)?wak????1????#@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9]^>?Qi@I??j?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?ࡄ?????~???N??M??E???!]?????@	!       "	!       *	!       2$	?????0@ ?!?5???	h"l?@!?>d@:	!       B	!       J$	?[????????l?????|?r???!6[y??$??R	!       Z$	?[????????l?????|?r???!6[y??$??b	!       JCPU_ONLYY]^>?Qi@b q??j?W@
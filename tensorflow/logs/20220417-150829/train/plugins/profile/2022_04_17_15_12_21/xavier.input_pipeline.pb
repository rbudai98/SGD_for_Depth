$	?^??}$@ե???[|
??\"@!S{m??%@$	?~???w@?¾??@?Y?䈣??!NJ??$@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1*;??.r$@6ɏ?+??A>{.S?@ @YS?1?#??r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1r?CQ??#@VҊo(???A?a???l@Y?-@?j??r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails15??.=#@&qVDM???A?b??^?@Y????]???r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??s|??$@ǜg?K? @A?WY?G @YJ
,?)??r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1wg???#@?Y?N @A?I?U?@Y%???w??r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1bhur?B#@cD??2??AGsd嗑@Y?%??6??r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	Xs?`??#@t?^?? @A??!?kN@Y?D?[????r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
?1w-!?$@??^?s?@A?D(b?@Y??^?2???r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?6?[?$@??PMI? @Aŭ???@Y?(%????r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?΅?^?#@???
x @A??捓?@Y?#bJ$Q??r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1[|
??\"@?B?? @A8??@M@Yg??????r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1e73??h%@*???
?@A-"??P@Y<l"38??r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?R??F?"@?S ?g0 @A???:Tc@YG ^?/???r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1ҍ????#@?t?_?
??A+0du?'@Y??(@???r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1ő"??$@?stT @Ax??Dgy@Yc?????r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1S{m??%@9b->  @A????E @Y???}???r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1}v?uż#@?<?+J???A?u?X?n@Y?A?<????r	train 517*	?rh?ͤ?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat????c@!?? ??A@)?L!u[@1?7X???>@:Preprocessing2T
Iterator::Root::ParallelMapV2}??z?V??!9?J???3@)}??z?V??19?J???3@:Preprocessing2E
Iterator::Root?h?wa@!??????A@)\?tYLl??1Id??GE0@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceޑ??????!I^?td)@)ޑ??????1I^?td)@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipܻ}?@!??	7 P@)?3?????1?{?j?!@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate:?<c_???!????ǝ1@).u?׃I??1?Y/i??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???B??!?1G}.?@)???B??1?1G}.?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?PO????!??!?7?4@).S??i??1?
???
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t20.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9o?J?y@I?Q??ohW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	s??? @Z]?79???&qVDM???!*???
?@	!       "	!       *	!       2$	???J	@?Uо??8??@M@!?WY?G @:	!       B	!       J$	@???m????Ŵ ??J
,?)??!?(%????R	!       Z$	@???m????Ŵ ??J
,?)??!?(%????b	!       JCPU_ONLYYo?J?y@b q?Q??ohW@
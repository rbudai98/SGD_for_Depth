$	&??"?#@h?3????W?????!@!~T?~O?%@$	?|ZlyV@??I?I?@d??܄??!mܙSZ#@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1qVDM?y%@z?]?z?@A~?Az?| @Y??aMeQ??r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?4?\??#@J&?v?@A??ؙB@Y???????r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1Q3????#@?q?Z|???A?G??5?@Y?I`s???r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1VF#?W?"@wN?@?C @A?	??@Y?@?????r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1W?????!@e?????A?? ?>@Yg??}q???r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1!???$@}?%?o??A???q? @Y7??VBw??r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	Q?y?#@Y??Z????AT?YO?@Y?n?????r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
?ؘ??#@=?1Xq??A??`???@YCUL??p??r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??Tka#@;?*?X??A-???@Y(?H0?L??r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?5???:%@y?@A[??	?@YG?	1????r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1~T?~O?%@eU???*@A??쿞@Y?"??$??r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1? #??#@???|@A??????@Ym???L??r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??fԄ#@???)W??A?z??9@YL???<???r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1qZ???@#@%u????A?/EH?@Y?@gҦj??r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1K?rJ?#@1Xr??A9
?@Y????????r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1<?H??z"@?9w?^???A}?%?o@Y??ᔹ???r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1/ܹ0ҳ#@??e?	??APqx?l@Y??u?X???r	train 517*	???K?5?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatiq?0'h@!M?`k8C@)G8-x?w@1U?T?3A@:Preprocessing2T
Iterator::Root::ParallelMapV2\?????!?????2@)\?????1?????2@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceqN`???!"%s??)@)qN`???1"%s??)@:Preprocessing2E
Iterator::Root??5&? @!=5· ?@)AaP??d??1?B-??(@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip1?????@!??{??Q@)?tw?y??1@dij>?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate3???V???!uU? 
4@)?FY?????1?3n>?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???[??!?RtA?@)???[??1?RtA?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap???????!????N7@)=+i?7??1????'
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t20.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???u@I0 ????W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??9????bEќK???=?1Xq??!eU???*@	!       "	!       *	!       2$	?w?o?|@	U0?!??}?%?o@!~?Az?| @:	!       B	!       J$	Ƿ????!Ƒ?????@?????!?"??$??R	!       Z$	Ƿ????!Ƒ?????@?????!?"??$??b	!       JCPU_ONLYY???u@b q0 ????W@
$	??o!??(@xuҝx	??R?U???%@!O?`??,@$	>???$@jI???@?t?:@3@!=???!4@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1M1AG?(@/?.?(@AW??U= @YN?G?????r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?????+@?8?#+?@A?(A?'#@Y
J?ʽ@??r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails14??O?*@Q?O?I??AUh ???#@YR?U??; @r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??????'@?%???O@Ap?71$g@Yo?$????r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
J?ʽ?&@?ɧ?v??AΉ=??m!@Y?#?????r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?e3??)@$}Z%@A??^b,?!@Y؂?C ??r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	O?`??,@YM?]?@A??cZ?&$@Y??8?j???r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
G???)@e?f	@A?.??"?@Y}?F???r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?????)@:??ll@A܁:??@Y\????@r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?????*@B	3m?*@Aa?????$@Y(Hlw???r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1EdX??(@????V@A??(??? @Y??O????r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1.????'@O?s?L@AO????@Y??J?.???r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?$y??k'@3?ۃ?@A?]i?@YI?V????r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?????&@??*P??@A??@?? @YN?w(
???r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1R?U???%@??@A??
??	@Yѭ??? ??r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??J??)@?/?P@A6?Ko?!@Y2W????r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?-?R\?&@؟??N?@A?bb?q?@YVDM?????r	train 517*	??ʁ?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatXc'?d@!?#F}u??@)d?8@1?,?l??<@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice#?	????!?!pB&3@)#?	????1?!pB&3@:Preprocessing2T
Iterator::Root::ParallelMapV2K??^b???!???<}2@)K??^b???1???<}2@:Preprocessing2E
Iterator::Root??2??@!^,???%@@)?_?+?[??1ǳ	B?+@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip6=((E?@!?i,??P@)???|?r??1???(?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap????5@!i`,?,?<@)?B?????1?#yoY?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate|?O?? @!~NEV?7@)R?U?????11?w@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??^??!o?o???@)??^??1o?o???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 10.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t21.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??f}??$@I?(Sp?aV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	O?s?L@????n??Q?O?I??!e?f	@	!       "	!       *	!       2$	?????? @??j??J??p?71$g@!a?????$@:	!       B	!       J$	q??Vs??????ۥ|??(Hlw???!\????@R	!       Z$	q??Vs??????ۥ|??(Hlw???!\????@b	!       JCPU_ONLYY??f}??$@b q?(Sp?aV@
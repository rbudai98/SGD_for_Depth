?&$	?J?da"@?E???????=Ab?k@!??8Qe&@$	?????@?L??????2sb???!?b??]@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1܂???g"@v3?'??A ??? @Y?PlMK??r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?l????!@w/??Q???A7?~$@Y?s}??r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?X"@??ص?} @A?????'@Y?j?=&R??r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1????)"@v?1<6??Alˀ??|@Y!?X4????r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??3??="@?h?"??A?7?q??@Y!v??y???r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??zܷ2#@h]??@o @AqqTn?6@Y????????r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	??'?)"@?fG??<??A?????@YU???N@??r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
ͭVcI"@???9#???A???iY@Y?h9?Cm??r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1Ʀ?B ?!@J)??????A?F???8@Y??_?????r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?=Ab?k@?8'0??Ab?c?@Y??K7?A??r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1e?fb2"@L7?A`???A]??ky?@YЀz3j???r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??e?O?"@?]M??Z??Ar3܀?@YÜ?M??r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??kB!@r??	???A?????@Yp??;???r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?Y??U1"@?
E????A??????@Y???a???r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??m?"@?ʾ+???ADܜJ?@Y?U??f???r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1_F????"@?w?7NJ??A?-y<@Y???I????r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??8Qe&@?A	3m?@A??O?m? @Y/?$???r	train 517*	4333??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?n?ʍ@!??U??l@@)???Z`O??1?K yK=@:Preprocessing2T
Iterator::Root::ParallelMapV2?^?????!p:a?1@)?^?????1p:a?1@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceQ29?3???!K??k0@)Q29?3???1K??k0@:Preprocessing2E
Iterator::Root????w?@!?{??@@)=}?????1?w{rA/@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?׺??@!pBp??P@).??H??1>?+4"@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateR?U?????!?>??p6@)s?????1PW>V?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??ɩ?a??!?m@)??ɩ?a??1?m@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapHk:!t??!L?H(??8@)???=z???1p?O??}@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?F?i??@I?e???RX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?iY?S???%m????w/??Q???!?A	3m?@	!       "	!       *	!       2$	?F?a??@?ۇ????b?c?@!??O?m? @:	!       B	!       J$	?????????|?V?J??Ѐz3j???!U???N@??R	!       Z$	?????????|?V?J??Ѐz3j???!U???N@??b	!       JCPU_ONLYY?F?i??@b q?e???RX@Y      Y@q??!y@"?
both?Your program is POTENTIALLY input-bound because 19.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 
?&$	?[???^"@Tl2??e??u??<? @!pD??k?#@$	ƃ?t?@??? ;???to???!X??-@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?I}Y?Q"@u?b?T?@A/??C@YZ?X"???r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1pD??k?#@???"????A3NCT?@Y???????r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1M-[??4!@????? @AՕ??<?@Yq????V??r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1dZ????#@??/?x0 @A?????Y@Y8???n???r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1c{-??i"@?(?????A?ӻx?^@Y??Q?d??r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??tBX"@r3܀ @Al??g??@Y?T?G????r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	u??<? @??F;nx??A=?E~??@YZf?????r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
??!!@T?*?gz??A£?#??@Y????P??r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?Q?dM#@иp $???Ae???@Y?K⬈??r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1n/?@!@????/v??A??$>w?@Y=??? !??r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1y?ՏMb"@c|??l{??AV?pA?@Y??????r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1-y<-O#@T?qs???Ad????q@Y?j???#??r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??;?!@???Д= @A֪]?@Y?S?????r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1]??7??"@?Bus?7??A????@Y?A??????r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?}(Ff!@??}?G??A?1???@Y0?[w?T??r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???t?<#@??A?L??A?׼???@Y. ?ҥ??r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?^Ӄ??"@ ???7??A????a?@Y?E? ??r	train 517*	$???K?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat? ?bG?@!?S??N?C@)??C5%?@1ڶ??(?@@:Preprocessing2T
Iterator::Root::ParallelMapV2?Go??\??!????0@)?Go??\??1????0@:Preprocessing2E
Iterator::RootZ??϶@!{9??1?@)???Q??1?H??r,@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceU??????!Q?0??)@)U??????1Q?0??)@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?@,?9?@!????3Q@)?w?-;???1|??oG@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?w?????!6O$-{?2@)????*4??14YSpV@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??jQ??!??|I.?@)??jQ??1??|I.?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap6?:(??!?ي&u?5@)?t<f?2??1?R4??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??gֻ@I???L!RX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	@?????????ۮ???T?*?gz??!u?b?T?@	!       "	!       *	!       2$	Ov??@@?e	?Y??Օ??<?@!3NCT?@:	!       B	!       J$	-~Xt???????n???A??????!??????R	!       Z$	-~Xt???????n???A??????!??????b	!       JCPU_ONLYY??gֻ@b q???L!RX@Y      Y@qϢ???3@"?
both?Your program is POTENTIALLY input-bound because 20.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
$	?	F?O?$@?{i?y'??? 5?l?!@!??bc^?*@$	qt??h"@???@<?????!>??]\?5@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??N?`?&@x??q?@A(?>??@Y????e??r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?b???($@
??O? @A?9A?@Y9??!???r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1W???xm'@?t<f?R@A?J&??@YC??U@r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??E|'?&@C???|m@A	4??i@Y?pt??n??r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1dY0?#@???WW%@A%<?ן@Y?5??D??r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???ԴK#@v??^???A[%X.@Y&7?????r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	)?QG?-%@}Yک????A??fc%? @Y??ꫫ??r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
v??҃#@{M
J@A?F??@Y?????X??r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???Z?S%@j?TQ?j@Aol?`?@YFzQ?_??r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???e?y$@L?1?=???A?ɐ@YJ???????r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?4`??q%@??90 @A?[Z?+@Y?????r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1ƣT??#@???.@Ar5?+-S@Y?H?[???r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??bc^?*@,???4	@Ayu?ٛ"@Y???s??r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1ȴ6???"@O=?බ @A???{??@Ya?ri|??r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?vۅ??"@?$??Z??Az?"n^@Y??-$???r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1? 5?l?!@"P??H @AA-?@Y?^)?G??r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1غ??L"@ۋh;?? @Ar???@Y??y ?|??r	train 517*	?l?????@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeata????/@!?q ?:?A@)?3??? @1??rU??>@:Preprocessing2T
Iterator::Root::ParallelMapV2ޮ??0??!z??tm/@)ޮ??0??1z??tm/@:Preprocessing2E
Iterator::RootG ^?/? @!B???@)??$W???1?ޮ?ؠ.@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??@?ȓ??!>x ?`O.@)??@?ȓ??1>x ?`O.@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??e?????!]9 ?n7@)?J?8??1{?Eu? @:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipJ@L?@!?/پB>Q@)`??9z??1[????@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*Ҧ??\??!??7???@)Ҧ??\??1??7???@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap}w+K4??!??~?ݲ:@)"???1???16
%??
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t20.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??@?"@IP?׿<?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	-{i?)@?#??????$??Z??!,???4	@	!       "	!       *	!       2$	?G?װ@:VnÌ???A-?@!yu?ٛ"@:	!       B	!       J$	???/????]>??????ꫫ??!C??U@R	!       Z$	???/????]>??????ꫫ??!C??U@b	!       JCPU_ONLYY??@?"@b qP?׿<?V@
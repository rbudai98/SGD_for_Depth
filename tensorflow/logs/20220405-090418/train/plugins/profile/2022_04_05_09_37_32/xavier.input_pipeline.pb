$	????!@
?'~?r??m ]lZy @!?p??#@$	?%???F@??G&?
???mS?{$??!lN???@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??P???"@?z0)?@A?p??|?@Y?-s?,??r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?F ^?7!@?Z?????A7R?H?=@Y?Nw?x???r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??kzPP"@?n?EE??A&P6??@Y?@gҦ???r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails16?,?!@?H?"i7??Apa?xw@Y?????@??r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?p??#@?K??? @Aݶ?Q?@Y攀????r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?!?
??!@??l?%?@A?M?x@Y[(?????r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	??Gܮ!@:?m½???A??b)??@Y?"R?.??r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
l$	??!@???b @A?Fu:?5@Y"¿3??r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?el?f'"@M??΢w??A'"?
@Y??r?????r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1m ]lZy @?K?????A????=@YO??????r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1M!u;[!@?=?N????A??7?-?@YGɫs??r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1R???d!@ 7????Ae?I)?6@Y?#?G??r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???O!@|~!<???A	?/??@Y??Gq??r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?d??)9"@s+??X???A?x]/@Y4J??%??r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?s?^? @?????A??St$?@Yw稣???r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1G??R^?!@%Y???4??A9b->`@Yo?ŏ1??r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1;????? @E/?Xn???A??V_]@Yxρ???r	train 517*	`??"?µ@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??P?\|??!????A@)2t???1??uHT?@:Preprocessing2T
Iterator::Root::ParallelMapV2???$?[??!???Z2@)???$?[??1???Z2@:Preprocessing2E
Iterator::Rootw|?????!?%???NA@)\U?]???1N???B0@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??n-??!.aw%?}.@)??n-??1.aw%?}.@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipI,)w?#@!????XP@)r??Q????1??f?؇@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???
???!?P??[5@)? L????1I?Q??q@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?>??s???!? ??L8@)?=?????1?\ 3D?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*$}ZE??!?;?@)$}ZE??1?;?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?x?L?2@I:??k^X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	z???????V???E/?Xn???!?z0)?@	!       "	!       *	!       2$	R???o?@o{ ?v1????St$?@!ݶ?Q?@:	!       B	!       J$	Yw?/Ӛ????@Ĵ??-s?,??!4J??%??R	!       Z$	Yw?/Ӛ????@Ĵ??-s?,??!4J??%??b	!       JCPU_ONLYY?x?L?2@b q:??k^X@
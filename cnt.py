str= "[CLS] brown pelican chick. [CLS] animal. [CLS] wind/sup-board. [CLS] pool. [CLS] the outstretched arm of a man holding wii controllers. [CLS] riding horse standing in a mar \
ket place. [CLS] power line insulator. [CLS] young man in green eating a doughnut. [CLS] man in the blue shirt and shorts behind the young boy. [CLS] a woman in glasses and a white shirt holding \
 a pizza. [CLS] hut/tent. [CLS] minibus. [CLS] low-density smoke. [CLS] cng. [CLS] pickup truck. [CLS] tugboat. [CLS] great blue heron nest. [CLS] front loader/bulldozer. [CLS] t-bar. [CLS] a re \
d leather chair next to an end table with a box of tissues on it. [CLS] tank car. [CLS] cable tower. [CLS] the plane that is in focus. [CLS] smoke. [CLS] white ibis nest. [CLS] airplane. [CLS] s \
ailboat. [CLS] black skimmer flying. [CLS] a drink with ice in it. [CLS] trailer. [CLS] great blue heron egg. [CLS] bicycle. [CLS] large vehicle. [CLS] tern. [CLS] other bird. [CLS] vehicle lot. \
 [CLS] landslide. [CLS] small aircraft. [CLS] passenger vehicle. [CLS] swimmer. [CLS] truck w/box. [CLS] bridge. [CLS] excavator. [CLS] waterbuck. [CLS] small car. [CLS] manual van. [CLS] wind t \
urbine. [CLS] reach stacker. [CLS] utility truck. [CLS] older man in a white shirt and black pants about to throw a frisbee. [CLS] ground grader.', '[CLS] a cow with its head up in the air. [CLS] person. [CLS] shipping container lot. [CLS] passenger vehicle. [CLS] oil storage tank. [CLS] brown pelican chick. [CLS] brown pelican - wings spread. [CLS] fairway. [CLS] front loader/bulldoze \
r. [CLS] the bicycle that is closest to the camera. [CLS] sailboat. [CLS] mobile crane. [CLS] stressed potato plant. [CLS] truck w/box. [CLS] laughing gull juvenile. [CLS] white ibis juvenile. [CLS] kayak. [CLS] a zebra standing in the lead of three other zebras. [CLS] helipad. [CLS] damage. [CLS] cattle. [CLS] black skimmer adult. [CLS] black crowned night heron adult. [CLS] waterbuck \
. [CLS] aircraft hangar. [CLS] helicopter. [CLS] crabmeat in a box with other vegetables. [CLS] bunker. [CLS] a college student checking her smart phone while hanging out in a study group. [CLS] \
 a woman in glasses and a white shirt holding a pizza. [CLS] sushi rolls with white rice. [CLS] excavator. [CLS] the outstretched arm of a man holding wii controllers. [CLS] smoke. [CLS] a perso \
n in a brown dog costume. [CLS] older man in a white shirt and black pants about to throw a frisbee. [CLS] mid-density smoke. [CLS] dirt. [CLS] reddish egret adult. [CLS] power line plate. [CLS] \
 palm. [CLS] trailer. [CLS] a blue - headed bird looking to the right. [CLS] american avocet adult. [CLS] barge.', '[CLS] swimmer. [CLS] car. [CLS] a bluish - gray lazy - boy reclining chair. [CLS] tricolored heron adult. [CLS] the man has on a dark shirt and no hat. [CLS] well. [CLS] mixed tern flying. [CLS] low-density smoke. [CLS] prefabricated house. [CLS] lightly damaged tree. [CLS] surfboard. [CLS] american oystercatcher. [CLS] person. [CLS] the motorcycle with the person sitting on it. [CLS] healthy potato plant. [CLS] container ship. [CLS] goat. [CLS] utility truck. [CLS] cycle. [CLS] yacht. [CLS] laughing gull juvenile. [CLS] tricycle. [CLS] small vehicle. [CLS] container crane. [CLS] the male in the middle. [CLS] the top of a 3 - tier cake. [CLS] cng. [CLS] bicycle. [CLS] pedestrian. [CLS] power line tower. [CLS] older man in a white shirt and black pants about to throw a frisbee. [CLS] a blue raft with a mans feet spread across the raft. [CLS] s \
traddle carrier. [CLS] stressed potato plant. [CLS] mixed tern adult. [CLS] a man driving a sporty street bike. [CLS] great blue heron adult. [CLS] white ibis juvenile. [CLS] bridge. [CLS] kayak \
. [CLS] boat. [CLS] manual van. [CLS] cable tower. [CLS] van. [CLS] haul truck. [CLS] the pa keyboard on the right , in front. [CLS] a zebra standing in the lead of three other zebras. [CLS] car \
go truck. [CLS] animal. [CLS] cattle egret flying. [CLS] great blue heron juvenile. [CLS] cultivation mesh cage."

str_by_dot = str.split(".")
for s in str_by_dot:
   if '[CLS]' not in s:
      print(s)
print(len(str_by_dot)) 

str_by_cls = str.split("[CLS]")
print(len(str_by_cls))


LISTA = [ 101,  1037, 29145,  3061,  1999,  1996, 
 2599,  1997,  2093,  2060,                                                                                                                                                                       
         29145,  2015,  1012,   101,  7965,  3392,  1012,   101,  7125,  1012,                                                                                                                    
           101,  4189,  4576,  1012,   101,  4654,  3540, 22879,  2953,  1012,                                                                                                                    
           101,  6174, 17980,  1012,   101, 10905,  4817,  1012,   101,  2829,                                                                                                                    
         21877, 19341,  2078,  1011,  4777,  3659,  1012,   101,  5870, 19739,                                                                                                                    
          3363,  4639,  1012,   101,  9710,  4744,  1012,   101,  2829, 21877,                                                                                                                    
         19341,  2078,  1999,  3462,  1012,   101,  6636,  4946,  1012,   101,                                                                                                                    
         13555,  1012,   101,  1037,  2630,  1011,  3753,  4743,  2559,  2000,                                                                                                                    
          1996,  2157,  1012,   101,  4049,  1012,   101, 21154, 11937, 21218,                                                                                                                    
          2015,  1012,   101,  2450,  2006,  7997,  2006,  2157,  4147,  2304,                                                                                                                    
          2327,  1998,  2304,  6045,  1012,   101, 12187,  1012,   101,  5527,                                                                                                                    
          4951,  1012,   101,  2307,  2630, 22914,  3909,  1012,   101,  1996,                                                                                                                    
         21059,  2849,  1997,  1037,  2158,  3173, 16568, 21257,  1012,   101,                                                                                                                    
          1996,  3756, 14757,  2006,  1996, 13972,  1005,  1055,  2157,  1012,                                                                                                                    
           101,  5402,  1012,   101,  2317, 22822,  8458, 14182,  1041, 17603,                                                                                                                    
          2102,  4639,  1012,   101, 10165,  1012,   101, 13987,  1012,   101,                                                                                                                    
         11669,  1013, 11385,  1012,   101, 20148,  1012,   101,  1037,  2158,                                                                                                                    
          4439,  1037,  4368,  2100,  2395,  7997,  1012,   101, 11661,  2911,                                                                                                                    
          1012,   101, 13233, 14557,  3269,  1012,   101,  1996, 10165,  2008,                                                                                                                    
          2003,  7541,  2000,  1996,  4950,  1012,   101,  4111,  1012,   101,                                                                                                                    
         22091,  5582,  1011, 13012, 23490,  1012,   101,  4744,  1059,  1013,                                                                                                                    
          6381,  1012,   101, 14175,  6277,  1012,   101,  2911,  1012,   101,                                                                                                                    
          2829, 21877, 19341,  2078, 11799,  1012,   101, 20981,  1041, 17603,                                                                                                                    
          2102,  1012,   101,  4316,  2843,  1012,   101,  2373,  2240,  5127,                                                                                                                    
          1012,   101,  2307,  2630, 22914,  8288,  1012,   101, 14662,  1012,                                                                                                                    
           101, 14182,  1041, 17603,  2102, 14556]

print(LISTA.count(101))
print(LISTA.count(1012))

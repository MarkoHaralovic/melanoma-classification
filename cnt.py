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


LISTA = [  101,  2829, 21877, 19341,  2078, 14556, 
 1012,   101,  4111,  1012,                                                                                                                                                                       
           101,  3612,  1013, 10514,  2361,  1011,  2604,  1012,   101,  4770,                                                                                                                    
          1012,   101,  1996, 21059,  2849,  1997,  1037,  2158,  3173, 16568,                                                                                                                    
         21257,  1012,   101,  5559,  3586,  3061,  1999,  1037,  3006,  2173,                                                                                                                    
          1012,   101,  2373,  2240, 16021, 20350,  1012,   101,  2402,  2158,                                                                                                                    
          1999,  2665,  5983,  1037, 23126, 24072,  1012,   101,  2158,  1999,                                                                                                                    
          1996,  2630,  3797,  1998,  9132,  2369,  1996,  2402,  2879,  1012,                                                                                                                    
           101,  1037,  2450,  1999,  7877,  1998,  1037,  2317,  3797,  3173,                                                                                                                    
          1037, 10733,  1012,   101, 12570,  1013,  9311,  1012,   101,  7163,                                                                                                                    
          8286,  1012,   101,  2659,  1011,  4304,  5610,  1012,   101, 27166,                                                                                                                    
          2290,  1012,   101, 15373,  4744,  1012,   101, 12888, 11975,  1012,                                                                                                                    
           101,  2307,  2630, 22914,  9089,  1012,   101,  2392,  7170,  2121,                                                                                                                    
          1013,  7087,  3527,  6290,  1012,   101,  1056,  1011,  3347,  1012,                                                                                                                    
           101,  1037,  2417,  5898,  3242,  2279,  2000,  2019,  2203,  2795,                                                                                                                    
          2007,  1037,  3482,  1997, 14095,  2006,  2009,  1012,   101,  4951,                                                                                                                    
          2482,  1012,   101,  5830,  3578,  1012,   101,  1996,  4946,  2008,                                                                                                                    
          2003,  1999,  3579,  1012,   101,  5610,  1012,   101,  2317, 21307,                                                                                                                    
          2483,  9089,  1012,   101, 13297,  1012,   101,  9498, 11975,  1012,
           101,  2304,  8301, 15810,  3909,  1012,   101,  1037,  4392,  2007,
          3256,  1999,  2009,  1012,   101,  9117,  1012,   101,  2307,  2630,
         22914,  8288,  1012,   101, 10165,  1012,   101,  2312,  4316,  1012,
           101, 28774,  2078,  1012,   101,  2060,  4743,  1012,   101,  4316,
          2843,  1012,   101, 20148,  1012,   101,  2235,  2948,  1012,   101,
          4628,  4316,  1012,   101, 13361,  1012,   101,  4744,  1059,  1013,
          3482,  1012,   101,  2958,  1012,   101,  4654,  3540, 22879,  2953,
          1012,   101,  2300, 24204,  1012,   101]

print(LISTA.count(101))
print(LISTA.count(1012))

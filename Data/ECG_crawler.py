# Library
import urllib 
# Function

cutcut = '<option selected="selected" value="100">100</option> <option value="101">101</option> <option value="102">102</option> <option value="103">103</option> <option value="104">104</option> <option value="105">105</option> <option value="106">106</option> <option value="107">107</option> <option value="108">108</option> <option value="109">109</option> <option value="111">111</option> <option value="112">112</option> <option value="113">113</option> <option value="114">114</option> <option value="115">115</option> <option value="116">116</option> <option value="117">117</option> <option value="118">118</option> <option value="119">119</option> <option value="121">121</option> <option value="122">122</option> <option value="123">123</option> <option value="124">124</option> <option value="200">200</option> <option value="201">201</option> <option value="202">202</option> <option value="203">203</option> <option value="205">205</option> <option value="207">207</option> <option value="208">208</option> <option value="209">209</option> <option value="210">210</option> <option value="212">212</option> <option value="213">213</option> <option value="214">214</option> <option value="215">215</option> <option value="217">217</option> <option value="219">219</option> <option value="220">220</option> <option value="221">221</option> <option value="222">222</option> <option value="223">223</option> <option value="228">228</option> <option value="230">230</option> <option value="231">231</option> <option value="232">232</option> <option value="233">233</option> <option value="234">234</option>'
cutcut = cutcut.split("</option>")
data_index = []
for i in range(len(cutcut)): 
    if len(cutcut[i][-3:]) > 1: 
        data_index.append(cutcut[i][-3:])

for data_num in data_index:	
    url_mfile = "http://www.physionet.org/atm/mitdb/" + data_num + "/atr/0/e/export/matlab/" + data_num + "m.mat"
    url_info = "http://www.physionet.org/atm/mitdb/" + data_num + "/atr/0/e/export/matlab/" +  data_num +"m.info"
    url_header = "http://www.physionet.org/atm/mitdb/" + data_num + "/atr/0/e/export/matlab/" + data_num + "m.hea"
    url_anno = "http://www.physionet.org/atm/mitdb/"+ data_num + "/atr/0/e/rdann/annotations.txt"

    print url_mfile
    print url_info
    print url_header
    print url_anno
    print ""
    
    urllib.urlretrieve(url_mfile, data_num+"_file.mat")
    urllib.urlretrieve(url_info, data_num+"_info.txt")
    urllib.urlretrieve(url_header, data_num+"_header.txt")
    urllib.urlretrieve(url_anno, data_num + "_anno.txt")

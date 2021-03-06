import re
import json
import csv
import math
import string

class Venue():
    
    def __init__(self, keyValMap, keyList):
        self.keyValMap = {}
        self.keyList = keyList 
        self.keyList.sort() ## sort list to make sure order is preserverd
        for k,v in keyValMap.iteritems():
            self.keyValMap[k] = self.clean(k,v) 


    def clean(self, k, v):
        """
            k: key for venue attribute
            v: value
            return: clean value for v depending on type of key
        """
        if k == 'phone':
            if v is not None:
                _v = re.sub("[^0-9]", "", v)
                return _v
            return v

        if k == "street_address":
##            return v
            _v = string.capwords(v)
            
            _v = re.sub(r'(W\.|West|W )', 'W ', _v)
            
            _v = re.sub(r'(E\.|East|E )', 'E ', _v)

            _v = re.sub(r'(S\.|South|S )', 'S ', _v)

            _v = re.sub(r'(N\.|North|N )', 'N ', _v)
            
            _v = re.sub(r'(Ne\.|Northeast|Ne )', 'NE ', _v)
            
            _v = re.sub(r'(Se\.|Southeast|Se )', 'SE ', _v)
            
            _v = re.sub(r'(Blvd\.|Boulevard|Blvd)', 'Blvd', _v)
            
            _v = re.sub(r'(Aly\.|Aly|Alley)', 'Aly', _v)
            
            _v = re.sub(r'(Sq\.|Sq|Square)', 'Sq', _v)
            
            _v = re.sub(r'(St\.|Street|St)', 'St', _v)
            
            _v = re.sub(r'(Place\.|Pl |Place)', 'Pl ', _v)
            
            _v = re.sub(r'(Ave\.|Av\.|Av|Ave|Avenue)', 'Ave', _v)
            
            _v = re.sub(r'(Plz\.|Plz|Plaza)', 'Plz', _v)

            _v = re.sub(r'(One\.|One)', '1', _v)
            _v = re.sub(r'(Two\.|Two)', '2', _v)
            _v = re.sub(r'(Three\.|Three)', '3', _v)
            _v = re.sub(r'(Four\.|Four)', '4', _v)
            _v = re.sub(r'(Five\.|Five)', '5', _v)
            _v = re.sub(r'(Six\.|Six)', '6', _v)
            _v = re.sub(r'(Seven\.|Seven)', '7', _v)
            _v = re.sub(r'(Eight\.|Eight)', '8', _v)
            _v = re.sub(r'(Nine\.|Nine)', '9', _v)
            
            
            return _v

        if k == 'website':
            _v = v.lower()
            _v = re.sub(r'(http://)', '', _v)           
            _v = re.sub(r'www\.', '', _v)            
            return _v 

        
        return v    



class GenericMiner():
    def __init__(self):
        pass

    def train(self, tr1, tr2, answers, miters, lrate):
        raise Exception("function not implemented for super class")

    def train_from_file(self, tr1_file, tr2_file, answers_file):
        """
            assumes tr1_file and tr2_file are JSON
            answers_file is .csv
        """
        tr1 = self.manifest_venues(tr1_file)
        tr2 = self.manifest_venues(tr2_file)
        answers = self.generate_matches_map(answers_file, "locu_id", "foursquare_id")
        self.train(tr1, tr2, answers)


    def classify_from_file(self, data1_file, data2_file):
        """
            assumes data1_file and data2_file are JSON files
            return: matches.csv file of all the matches found by perceptron
        """
        data1 = self.manifest_venues(data1_file)
        data2 = self.manifest_venues(data2_file)
        return self.classify(data1, data2)

        ## TODO: finish
        

    def manifest_venues(self,filename):
        """
            Create Venue objects from JSON file
        """
        venues = []
        f = open(filename)
        data = json.load(f)
        for i in xrange(len(data)):
            keyValMap = {}
            keyList = []
            for k,v in data[i].iteritems():
                keyValMap[str(k)] = v
                keyList.append(str(k))
            venue = Venue(keyValMap, keyList)
            venues.append(venue)
        return venues

    def generate_matches_map(self, filename, id1_name, id2_name ):
        """
            Create HashSet of venue ids that match
        """
        res1_2 = {}
        res2_1 = {}
        matches = csv.DictReader(open(filename), "rU")
        for m in matches:
            if m['r'] != id1_name and m['U'] != id2_name:
                id1 = m['r']
                id2 = m['U']                
                res1_2[id1] = id2
                res2_1[id2] = id1
        return res1_2, res2_1


    def get_feature_vector(self, venue1, venue2):
        """
            creates feature vector of properties of Venues
        """
        ## think about latitude and longitude combined maybe
        
##        feature_score_vector = []
##        for k in venue1.keyList:
##            v1 = venue1.keyValMap[k]
##            v2 = venue2.keyValMap[k]
##            feature_score_vector.append(self.get_score(keyList, k,v1,v2))

        return self.get_score_vector(venue1, venue2)

        


    def get_score_vector(self, venue1, venue2):
        """
            return score vector for two venues compared
        """
        vector = []

        v1Map = venue1.keyValMap
        v2Map = venue2.keyValMap
        
        ## country feature
        c1 = v1Map["country"].lower()
        c2 = v2Map["country"].lower()
        if c1 == c2:
            vector.append(1)
        else:
            vector.append(0)


        ## second feature: latitude and longitude combined        
        lat1 = v1Map["latitude"]
        lat2 = v2Map["latitude"]
        long1= v1Map["longitude"]
        long2= v2Map["longitude"]
        if(lat1 is None or lat2 is None or long1 is None or long2 is None):
            vector.append(0)
        else:
            delta_lat = abs(lat2 - lat1)
            delta_long = abs(long2 - long1)

            METERS_THRESHOLD = 20
            meters = delta_lat *  40008000/float(360)
            if (meters < METERS_THRESHOLD):
                delLong_max = METERS_THRESHOLD * 360 * float(1/ math.cos(lat1*math.pi/180)) * float(1)/ 40075160
                if delta_long < delLong_max:
                    vector.append(5)
                else:
                    vector.append(0)
            else: vector.append(0)
            
        ## locality
        loc1 = v1Map["locality"].lower()
        loc2 = v2Map["locality"].lower()
        if loc1 == loc2:
            vector.append(1)
        else:
            vector.append(-1)

        ## name:
        n1 = v1Map["name"].lower()
        n2 = v2Map["name"].lower()
        if self.levenshtein_distance(n1, n2) < 4:
            vector.append(5)
        else: vector.append(0)

        #phone
        ph1 = v1Map["phone"]
        ph2 = v2Map["phone"]
#	if ph1 or ph2 == None:
#	    vector.append(1)
        if ph1 == ph2: vector.append(5)
        else: vector.append(0)

        #postal code and address combined
        zip1 = v1Map["postal_code"]
        zip2 = v2Map["postal_code"]
        street_addr1 = v1Map["street_address"]
        street_addr2 = v2Map["street_address"]

        if street_addr1 == street_addr2:
            if zip1 == zip2:
                vector.append(5)
            elif zip1 == None or zip2 == None:
                vector.append(1)
            else:
                vector.append(-4)
        else:
            vector.append(-5)
            
            
        
##        if self.levenshtein_distance(street_addr1,street_addr2) < 4:
##            if zip1 == zip2:
##                vector.append(5)
##            elif zip1 == None or zip2 == None:
##                vector.append(1)
##            else:
##                vector.append(-3)
##        else:
##            vector.append(-5)

        #region
        reg1 = v1Map["region"].lower()
        reg2 = v2Map["region"].lower()
        if reg1 == reg2: vector.append(1)
        else: vector.append(0)


        #website
        web1 = v1Map["website"].lower()
        web2 = v2Map["website"].lower()
        if web1 == web2:
            vector.append(5) 
	
	elif web1 == None or web2 == None:
	    vector.append(0)	
      
        else:
            vector.append(-5)

        ##done
        return vector
            
    
    def false_negative_check(self, venue1, venue2):
        """
            returns true if this method thinks that these are false negatives - i.e should match
        """
        v1Map = venue1.keyValMap
        v2Map = venue2.keyValMap

        phoneEq = False
        streetEq = False
        nameEq = False
        

        if v1Map["phone"] is not None and v1Map["phone"] != "":
            if v1Map["phone"] == v2Map["phone"]:
                phoneEq = True
        if v1Map["street_address"] is not None and v1Map["street_address"] != "":
            if v1Map["street_address"] == v2Map["street_address"]:
                streetEq = True 
        
        if v1Map["name"] is not None and v1Map["name"] != "":
            if v1Map["name"] == v2Map["name"]:
                nameEq = True


        
        return ((phoneEq and streetEq) or (phoneEq and nameEq) or (nameEq and phoneEq))
        

##        lat1 = v1Map["latitude"]
##        lat2 = v2Map["latitude"]
##        long1= v1Map["longitude"]
##        long2= v2Map["longitude"]


##        if(lat1 is not None and lat2 is not None and long2 is not None and long2 is not None):
##            delta_lat = abs(lat2 - lat1)
##            delta_long = abs(long2 - long1)
##
##            METERS_THRESHOLD = 20
##            meters = delta_lat *  40008000/float(360)
##            if (meters < METERS_THRESHOLD):
##                delLong_max = METERS_THRESHOLD * 360 * float(1/ math.cos(lat1*math.pi/180)) * float(1)/ 40075160
##                if delta_long < delLong_max:
##                    return True
##                else:
##                    return False
##            else: return False
##        else:
##            return False

                        
##        return False
            

    
    def levenshtein_distance(self, st1, st2):
        """ source: en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance
            implements Levenshtein distance algorithm
        """
        if len(st1) > len(st2):
            return self.levenshtein_distance(st2, st1)

        if len(st2) == 0: return len(st2)

        prev_row = xrange(len(st2) + 1)
        for i, char1 in enumerate(st1):
            curr_row = [i+1]
            for j, char2 in enumerate(st2):
                insertNum = prev_row[j+1] + 1
                deleteNum = curr_row[j] + 1
                subsNum = prev_row[j] + (char1 != char2)
                curr_row.append(min(insertNum, deleteNum, subsNum))
            prev_row = curr_row

        return prev_row[-1]




    def jaccard(self, str1, str2):
        str1 = set(str1)
        str2 = set(str2)          
        c = str1.intersection(str2)
        return 1 - (float(len(c)) / (len(str1) + len(str2) - len(c)))
        


    def dot_product(self, v1, v2):
        """
            Simply returns dot product for two vectors v1 and v2.
            requires v1 and v2 to have same dimension / length
        """
        assert len(v1) == len(v2)
        res = sum([v1[i]*v2[i] for i in xrange(len(v1))])
        return res
     
            
            



class PerceptronMiner(GenericMiner):
    def __init__(self):
        GenericMiner.__init__(self)
        self.attributeScores = []
        self.weights = None        

    def train_with_miters_lrate(self, tr1, tr2, answers, miters, lrate):
        """
            tr1: training set 1 - list of Venues
            tr2: training set 2 - list of Venues
            answers: two dictionaries res1_2: matches tr1 ids to tr2 ids, and res2_1:matches tr2 ids to tr1 ids 
            miters: max iterations for perceptron
            lrate: learning rate for algorithm
        """
        print "starting training"        
        res1_2 = answers[0]        

        ## fill venues from the two training sets in hash table
        print "fill venues in hashtable"
        
        _tr1 = {}
        _tr2 = {}
        for t1 in tr1:
            _t1 = t1.keyValMap
            _tr1[_t1["id"]]= t1

        for t2 in tr2:
            _t2 = t2.keyValMap
            _tr2[_t2["id"]]= t2

        print "done filling hashtables"        

        ##nasty trick to get size
        length = len(self.get_feature_vector(tr1[0], tr2[0]))
        self.weights = [0] + [0 for i in xrange(length)]
        print "weights initilized"
        print "weights = " , self.weights
        

        
        print "start perceptron training"
        ## perceptron training
        v1_v2_scores = {}
        for it in xrange(miters):
            for venue1 in tr1:
                for venue2 in tr2:
                    f_vector = []
                    if (venue1, venue2) not in v1_v2_scores:
                        f_vector = [1] + self.get_feature_vector(venue1, venue2)
                        v1_v2_scores[(venue1, venue2)] = f_vector
                    else:
                        f_vector = v1_v2_scores[(venue1, venue2)]
                    v1_id = venue1.keyValMap["id"]
                    v2_id = venue2.keyValMap["id"]
                    answer = -1 # no match assumption
                    if v1_id in res1_2:
                        match = res1_2[v1_id]
                        if match == v2_id:
                            answer = 1
                    margin = answer * self.dot_product(f_vector, self.weights)
                    if margin <= 0:
                        for i in xrange(len(self.weights)):
                            self.weights[i] += lrate * answer * f_vector[i]
                   
       

        print "done perceptron training"
        print "done training"

        return None        
                
        
    
    def train(self, tr1, tr2, answers):
        """
            tr1: training set 1
            tr2: training set 2
            answers : matching answers in training set
        """
        miters = 8 ## max iterations
        lrate = 0.75 ## learning rate

        self.train_with_miters_lrate(tr1, tr2, answers, miters, lrate)


    def classify(self, data1, data2):
        """
            data1: actual data set1
            data2: actual data set2  

            return matches, non_matches, result
            matches: list of (i,j) where i and j are ids of matches from data
            non_matches: list of (i,j) of all non-matches
            result: list of length data1*data2 where 1 stands for match, -1 otherwise
        """
        matches = []
        non_matches = []
        result = [0] * (len(data1)*len(data2))
        for venue1 in data1:
            for venue2 in data2:
                fv = [1] + self.get_feature_vector(venue1, venue2)
                dp = self.dot_product(self.weights, fv)
                id1, id2 = (venue1.keyValMap["id"], venue2.keyValMap["id"])
                if dp > 1 : #match
                    matches.append((id1, id2))
                else:
                    isFn = self.false_negative_check( venue1, venue2)
                    if(isFn): matches.append((id1,id2))
                    else: non_matches.append( (id1, id2))

        return matches, non_matches, result



if __name__ == "__main__":
    pMiner = PerceptronMiner()
    pMiner.train_from_file("locu_train_hard.json", "foursquare_train_hard.json", "matches_train_hard.csv")
    print "weights = " , pMiner.weights

    matches, non_matches, result = pMiner.classify_from_file("locu_train_hard.json", "foursquare_train_hard.json")
    res1_2,res2_1 = pMiner.generate_matches_map("matches_train_hard.csv", "locu_id", "foursquare_id")

    tp = 0 #true positives - correct matches
    fp = 0 #false positives - incorrect matches
    fn = 0 #false negatives - missing results

    print "matches:",len(matches)
    print "non_matches:", len(non_matches)
    print "truth:",  len(res1_2)
    
    for m in matches:
        if m[0] in res1_2:
            if m[1] == res1_2[m[0]]: ##correct match
                tp += 1
            else:
                print "fp = ", m
                fp += 1
        else:
            print "fp = ", m
            fp += 1
            

    for n in non_matches:        
        if n[0] in res1_2:
            if n[1] == res1_2[n[0]]:
                print n
                fn += 1

    print "TP=", tp
    print "FP=", fp
    print "FN=", fn

    recall = tp / (float(tp + fn))
    precision = tp /(tp + float(fp))
    print "recall:", recall
    print "prec:", precision
    print "f1Score:", 2*precision * recall / (float(recall+precision))
                      
    
    #matches, non_matches, result = pMiner.classify_from_file("locu_test_hard.json", "foursquare_test_hard.json")
    with open("matches_test.csv", 'w') as f:
        f.write("locu_id" + "," + "foursquare_id" + "\n")
        for m in matches:
            f.write(str(m[0]) + "," + str(m[1]) + "\n")
    

                        



        
        

    
    
    

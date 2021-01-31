import copy
import csv
import math

import numpy as np


def read_data(file):
    f = open(file, "rt", encoding="utf8")
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    data = [d for d in reader]
    return data


def splitData(allData, procent=0.7):
    trainData = []
    testData = []
    for d in allData:
        if np.random.rand() <= procent:
            trainData.append(d)
        else:
            testData.append(d)
    return trainData, testData


def RMSE(testData, Q, P, trainDataUsers, razred):
    x = [math.pow(float(rui) - (predictValue(Q, song, P, user, trainDataUsers, razred)), 2) for (user, song, rui) in
         testData]
    return math.sqrt(sum(x) / len(x))


def predictValue(Q, song_id, P, user_id, trainDataUsers, razred):
    if user_id in P.keys() and song_id in Q.keys():

        Tu_avg = np.zeros([1, razred.K]).flatten()
        Ti_avg = np.zeros([1, razred.K]).flatten()

        # uporabimo prvo metodo za pridobitev informacij iz tagov
        if razred.useTagInfo:
            if user_id in razred.userSongTags.keys() and song_id in razred.userSongTags[user_id]:
                for tag in razred.userSongTags[user_id][song_id]:
                    Ti_avg += razred.Y[tag]

                Ti_avg /= len(razred.userSongTags[user_id][song_id])

        # uporabimo drugo metodo za pridobitev informacij iz tagov
        if razred.useTagInfo2:
            if user_id in razred.userTags.keys():
                Tu = razred.userTags[user_id]
                lenTu = len(set(Tu))
                for el in set(Tu):
                    Tu_avg += Tu.count(el) * razred.Y[el]
                Tu_avg = Tu_avg / lenTu

            if song_id in razred.songTags.keys():
                Ti = razred.songTags[song_id]
                lenTi = len(set(Ti))
                for el in set(Ti):
                    Ti_avg += Ti.count(el) * razred.X[el]
                Ti_avg = Ti_avg / lenTi

        # če ne uporabljamo nobeno izmed metod za pridobivanje informacij iz tagov oz. tagov ni, dodajamo samo ničle
        ocena = (P[user_id] + Tu_avg).dot(Q[song_id] + Ti_avg)
    elif user_id not in P.keys() and song_id in Q.keys():
        # nov uporabnik, za izvajalca podamo povprečno oceno ostalih uporabnikov
        x = [trainDataUsers[x][song_id] for x in trainDataUsers if song_id in trainDataUsers[x].keys()]
        ocena = sum(x) / len(x)
    elif song_id not in Q.keys() and user_id in P.keys():
        # nov izvajalec, torej podamo povprečno oceno uporabnika
        d = trainDataUsers[user_id]
        ocena = sum(d.values()) / float(len(d))
    else:
        # nov izvajalec in nov uporabnik, podamo torej povprečje vseh ocen
        ocena = razred.avg  # povprečna ocena vseh učne množice user_artist_training.dat

    # porežemo ocene izven omejitev
    if ocena < 0:
        ocena = 0
    elif ocena > 10:
        ocena = 10

    # print("---------------------------------------------------------- ocena " + str(ocena))
    return ocena


# implementacija po clanku Matrix Factorization and Neighbor Based Algorithms for the Netflix Prize Problem- Gábor Takács,István Pilászy,Bottyán Németh,Domonkos Tikk
class RazcepMatrik():
    def __init__(self, trainData, testData, tags, K, useTagInfo=False, useTagInfo2=False):
        # trainData  - učni podatki, podatni kot seznam, kjer vsak element vsebuje uporabnika, izvajalca in oceno
        # testData   - testni podatki, podani v enakem formatu kot trainData
        # tags       - podatki o tagih
        # K          - parameter K
        # useTagInfo - boolean, ki določi ali uporabljamo prvo metodo, ki uporablja tage (True - uporabimo)
        # useTagInfo2 - boolean, ki določi ali uporabljamo drugo metodo, ki uporablja tage (True - uporabimo)
        # obe metodi se nista izkazali kot dobri, saj sta poslabšali rezultat
        # članek kjer sta metodi opisani http://ceur-ws.org/Vol-1245/cbrecsys2014-paper06.pdf
        # prva metoda - UserItemRelTags
        # druga metoda- TagGSVD++ -> napaka v implementaciji
        # Problem v slabši preformanci, je lahko napaka v implementaciji ali nezdružljivost z BRISMF metodo
        self.K = K
        self.testData = testData
        self.trainData = trainData
        self.tagsData = tags
        self.users = []  # seznam uporabnikov
        self.songs = []  # seznam pesmi
        self.userSongScore = {}
        self.predictions = []
        self.useTagInfo = useTagInfo
        self.useTagInfo2 = useTagInfo2
        for row in trainData:
            if row[0] not in self.users:
                self.users.append(row[0])
                self.userSongScore[row[0]] = {}
            if row[1] not in self.songs:
                self.songs.append((row[1]))
            self.userSongScore[row[0]][row[1]] = int(float(row[2]))
        self.avg, self.userGrades, self.songGrades = self.getAvgs()
        self.tags, self.userTags, self.songTags, self.userSongTags = self.getTags()
        self.P = {}
        self.Q = {}
        self.X = {}
        self.Y = {}
        for user in self.users:
            self.P[user] = np.random.uniform(-0.01,
                                             0.01,
                                             self.K)
            self.P[user][0] = 1

        for song in self.songs:
            self.Q[song] = np.random.uniform(-0.01,
                                             0.01,
                                             self.K)
            self.Q[song][1] = 1

        for tag in self.tags:
            self.X[tag] = np.random.uniform(-0.01,
                                            0.01,
                                            self.K)
            self.X[tag][0] = 1

            self.Y[tag] = np.random.uniform(-0.01,
                                            0.01,
                                            self.K)
            self.Y[tag][1] = 1

    # pridobimo povprečja
    def getAvgs(self):
        avgAll = 0
        userGrades = {}
        songGrades = {}
        for (user, song, grade) in self.trainData:
            grade = float(grade)
            avgAll += grade
            if user not in userGrades.keys():
                userGrades[user] = [grade]
            else:
                userGrades[user].append(grade)
            if song not in songGrades.keys():
                songGrades[song] = [grade]
            else:
                songGrades[song].append(grade)

        avgAll /= len(self.trainData)

        for x in userGrades:
            userGrades[x] = sum(userGrades[x]) / len(userGrades[x])
        for x in songGrades:
            songGrades[x] = sum(songGrades[x]) / len(songGrades[x])

        return avgAll, userGrades, songGrades

    # pridobimo podatke o tagih - vsi tagi, tagi posameznih izvajalcev, tagi posameznih uporabnikov, in tagi uporabnika za določenega izvajalca
    def getTags(self):
        userTags = {}
        songTags = {}
        userSongTags = {}
        allTags = []
        for (user, song, tag, day, month, year) in self.tagsData:
            if tag not in allTags:
                allTags.append(tag)
            if user not in userTags.keys():
                userTags[user] = [tag]
            else:
                userTags[user].append(tag)
            if song not in songTags.keys():
                songTags[song] = [tag]
            else:
                songTags[song].append(tag)

            if user not in userSongTags.keys():
                userSongTags[user] = {}

            if song not in userSongTags[user].keys():
                userSongTags[user][song] = []
            userSongTags[user][song].append(tag)
        return allTags, userTags, songTags, userSongTags

    def run(self, alpha=0.01, niq=0.005, nip=0.016):
        # print("Running...")

        T1, T2 = splitData(self.trainData)
        oldQ = oldP = 0
        maxRuns = 100
        # ponavljamo maksimalno maxRuns oziroma dokler je trenutna napoved boljša od prejšne (ocenimo z RMSE)
        nip = [0.0] + [nip] * (self.K - 1)
        alphap = [0.0] + [alpha] * (self.K - 1)
        niq = [niq, 0.0] + [niq] * (self.K - 2)
        alphaq = [alpha, 0.0] + [alpha] * (self.K - 2)

        for s in range(maxRuns):
            #print("Iteration: " + str(s))
            Q = copy.deepcopy(self.Q)
            P = copy.deepcopy(self.P)
            X = copy.deepcopy(self.X)
            Y = copy.deepcopy(self.Y)
            # print(self.songGrades)
            for (user, song, rui) in T1:
                Tu_avg = np.zeros([1, self.K]).flatten()
                Ti_avg = np.zeros([1, self.K]).flatten()

                # izvajamo prvo metodo uporabe informaciji iz tagov
                if self.useTagInfo:
                    if user in self.userSongTags.keys() and song in self.userSongTags[user]:
                        for tag in self.userSongTags[user][song]:
                            Ti_avg += Y[tag]

                        Ti_avg /= len(self.userSongTags[user][song])
                    Ti_avg = Ti_avg.flatten()

                # izvajamo drugo metodo uporabe informaciji iz tagov
                if self.useTagInfo2:
                    if user in self.userTags.keys():
                        Tu = self.userTags[user]
                        lenTu = len(set(Tu))
                        for el in set(Tu):
                            Tu_avg += Tu.count(el) * Y[el]
                        Tu_avg = Tu_avg / lenTu

                    if song in self.songTags.keys():
                        Ti = self.songTags[song]
                        lenTi = len(set(Ti))
                        for el in set(Ti):
                            Ti_avg += Ti.count(el) * X[el]
                        Ti_avg = Ti_avg / lenTi

                rui2 = (P[user] + Tu_avg).dot(Q[song] + Ti_avg)
                # self.avg+(float(rui)-self.songGrades[song]) + (float(rui)-self.userGrades[user])

                eui = float(rui) - rui2
                cf = P[user]
                mf = Q[song]

                for k in range(self.K):
                    P[user][k] += nip[k] * (eui * (mf[k] + Ti_avg[k]) - alphap[k] * cf[k])
                    Q[song][k] += niq[k] * (eui * (cf[k] + Tu_avg[k]) - alphaq[k] * mf[k])

                # gradientni sestop za prvo metodo z uporabo tagov
                if self.useTagInfo:
                    if user in self.userSongTags.keys() and song in self.userSongTags[user]:
                        for tag in self.userSongTags[user][song]:
                            for k in range(self.K):
                                Y[tag] += niq[k] * (
                                    eui * (1 / len(self.userSongTags[user][song])) * (cf[k]) - alphaq[k] * Y[tag][k])

                # gradientni sestop za drugo metodo z uporabo tagov
                if self.useTagInfo2:
                    for a in set(Tu):
                        for k in range(self.K):
                            X[a][k] = X[a][k] + nip[k] * (
                            eui * (Tu.count(a) / len(set(Tu))) * (mf[k] + Ti_avg[k]) - alphap[k] * X[a][k])

                    for b in set(Ti):
                        for k in range(self.K):
                            Y[b][k] = Y[b][k] + niq[k] * (
                            eui * (Ti.count(a) / len(set(Ti))) * (cf[k] + Tu_avg[k]) - alphaq[k] * Y[b][k])
            # print(RMSE(T2, Q, P)*100)
            # print("RMSE na učni množici: " + str(RMSE(T1, Q, P, self.userSongScore) * 100))
            # print(str(RMSE(T2, self.Q, self.P, self.userSongScore) * 100 - RMSE(T2, Q, P, self.userSongScore) * 100))
            # print(RMSE(T2, self.Q, self.P, self.userSongScore))
            if RMSE(T2, Q, P, self.userSongScore, self) < RMSE(T2, self.Q, self.P, self, self) or s < 2:
                # print("IMPROVED")
                oldP = 0
                oldQ = 0
                oldY = 0
                self.Q = Q
                self.P = P
                self.Y = Y
                self.X = X
            else:
                # poskrbimo, da lahko enkrat samkrat napovemo slabše, če naslednič ponovno napovemo slabše upoštevamo najboljšo napoved
                # drugače flaga oldP in oldQ resetiramo
                if oldP == 0 and oldQ == 0:
                    oldP = self.P
                    oldQ = self.Q
                    oldY = self.Y
                    oldX = self.X
                    self.Q = Q
                    self.P = P
                    self.Y = Y
                    self.X = X
                else:
                    R1 = RMSE(T2, oldQ, oldP, self.userSongScore, self)
                    R2 = RMSE(T2, self.Q, self.P, self.userSongScore, self)
                    R3 = RMSE(T2, Q, P, self.userSongScore, self)
                    if R1 == min([R1, R2, R3]):
                        self.Q = oldQ
                        self.P = oldP
                        self.Y = oldY
                        self.X = oldX
                    elif R3 == min([R1, R2, R3]):
                        self.Q = Q
                        self.P = P
                        self.X = X
                        self.Y = Y
                    # print("ENDED")
                    break
        for d in self.testData:
            ocena = predictValue(self.Q, d[1], self.P, d[0], self.userSongScore, self)
            self.predictions.append(ocena)


K = 8
# določimo ali poženemo na testnih podatkih ali pa poženemo za oddajo na server
testOnMyData = True
tags = read_data('user_taggedartists.dat')

if testOnMyData:
    allData = read_data('user_artists_training.dat')

    avgRMSE = 0
    noRepeat = 5

    for iRepeat in range(noRepeat):
        # print("Step number: " + str(iRepeat))
        trainData, testData = splitData(allData)
        r = RazcepMatrik(trainData, testData, tags, K)
        r.run()
        avgRMSE += RMSE(testData, r.Q, r.P, r.userSongScore, r) / noRepeat

    print("AVG. RMSE " + str(avgRMSE) )

    with open("testParam.txt", "a") as myfile:
        myfile.write(
            "\nAVG. RMSE " + str(avgRMSE))




else:
    testData = read_data('user_artists_test.dat')
    trainData = read_data('user_artists_training.dat')

    r = RazcepMatrik(trainData, testData, tags, K)

    r.run()
    f = open('results.txt', 'w')
    for res in r.predictions:
        f.write("%s\n" % res)

testCNaloga = True # določimo ali izpišemo napoved dobrih pesmi za uporabnika
if testCNaloga:
    trainData = read_data('user_artists_training.dat')
    allArtists = [x[1] for x in trainData]
    allArtists = list(set(allArtists))
    myData = [['1892', '2179', '5'],
              ['1892', '227', '5'],
              ['1892', '56', '6'],
              ['1892', '978', '9'],
              ['1892', '1347', '10'],
              ['1892', '14557', '4'],
              ['1892', '157', '7'],
              ['1892', '89', '4'],
              ['1892', '547', '8'],
              ['1892', '12621', '9'],
              ['1892', '1244', '7'],
              ['1892', '2236', '5'],
              ['1892', '2226', '4'],
              ['1892', '5986', '0'],
              ['1892', '966', '3'],
              ['1892', '65', '8'],
              ['1892', '321', '0'],
              ['1892', '5752', '2'],
              ['1892', '461', '1'],
              ['1892', '475', '8']
              ]

    trainData.extend(myData)
    songIds = [x[1] for x in myData]

    testData = []
    for artist in allArtists:
        if artist not in songIds:
            testData.append(['1892', artist])
    r = RazcepMatrik(trainData, testData, tags, K)
    r.run()

    results = np.array(r.predictions)
    top10i = results.argsort()[-10:][::-1]
    top10Results = []
    for el in top10i:
        top10Results.append(testData[el][1])

    artist = read_data('artists.dat')
    artistDict = {d[0]: d[1:] for d in artist}
    for i in range(10):
        print(artistDict[top10Results[i]])


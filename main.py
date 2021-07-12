import os
from sklearn import svm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import re # Veri ön işleme için gerekli.

nltk.download('wordnet')
nltk.download('stopwords')

# Dosyadan veri okuma işlemleri.
data = list()
for folder in os.listdir('bbc'):
    for file in os.listdir(os.path.join('./bbc', folder)):
        with open(os.path.join('./bbc', folder, file), errors='ignore') as text:
            words = text.read()
            data.append([words, folder])
df = pd.DataFrame(data, columns=['text', 'label'])

# Ön-İşleme için gerekli kütüphane.
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# Anlam bazlı kök bulma
lemmatizer = WordNetLemmatizer()
# Ek bazlı kök bulma
stemmer = PorterStemmer()

documents = []

#Veri ön-işleme ve verilerin anlamlı parçalara ayrılması(tokenization)
def preprocessing(data):
    documents = []
    for sen in range(0, len(data)):
        # \n kaldırma
        document = str(data[sen]).strip("\n")

        # Özel Karakterler kaldırılıyor.
        document = re.sub(r'\W', ' ', document)
        # Bütün tek karakterler kaldırılıyor.
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Baştaki tek harfli sözcükleri siler.
        document = re.sub(r'^[a-zA-Z]\s+', ' ', document)

        # Birden çok boşluk tek boşlukla değiştiriliyor
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Bütün yazı küçük harfe dönüştürüldü.
        document = document.lower()

        # Lemmatization
        document = document.split()
        document = [lemmatizer.lemmatize(word) for word in document]

        # Steming
        # document = [stemmer.stem(word) for word in document]

        document = ' '.join(document)
        documents.append(document)
    return documents

# Veri setimizi ön-işlemeye yolluyoruz.
documents = preprocessing(data)

# TF-IDF özellik matrisini oluşturuyoruz.
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
tf_idf_vector = tfidf_vectorizer.fit_transform(documents)

#Özellik sayısı
print(len(tfidf_vectorizer.get_feature_names()))

# Verileri etiketleri ile birlikte eğitim ve test olarak ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(tf_idf_vector, df.label)

# Model eğitiliyor.
svm = svm.LinearSVC()
svm.fit(X_train, y_train)

# Ayrılan test verileri sınıflandırılıyor.
print("SVM Accuracy ", svm.score(X_test, y_test))

# Test örneği.
ornek = list()
ornek.append("""

            It's an all-English Champions League final - but will it be Manchester City or
             Chelsea who will be crowned champions of Europe on Saturday?

Premier League champions City are looking to win the
 prestigious competition for the first time.

Chelsea, European champions in 2012, have beaten Pep
 Guardiola's side twice in the league and FA Cup in 2020-21.

Up to 16,500 people will be allowed inside Porto's
 Estadio do Dragao ground to watch.

Both sides have fully fit squads to choose from, although City midfielder Ilkay Gundogan
 looked like he took a minor knock in Friday's training session after a collision with Fernandinho.          

""")
# Gelen yeni veriyi ön-işlemeye yolluyoruz.
ornekProcessed = preprocessing(ornek)

ornekVeriTFIDFvektoru = tfidf_vectorizer.transform(ornekProcessed)

ornek_predicted = svm.predict(ornekVeriTFIDFvektoru)
print(ornek_predicted[0])

# Arayüz
from PyQt5 import QtCore, QtGui, QtWidgets

# PyQt5 dizaynı kullanılarak arayüz tasarımı.
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(220, 330, 341, 81))
        font = QtGui.QFont()
        font.setPointSize(17)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 440, 571, 91))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setOpenExternalLinks(False)
        self.label.setObjectName("label")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(20, 20, 751, 291))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.textEdit.setFont(font)
        self.textEdit.setDocumentTitle("")
        self.textEdit.setLineWrapColumnOrWidth(1)
        self.textEdit.setObjectName("textEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Bitirme Projesi"))
        self.pushButton.setText(_translate("MainWindow", "Kategori Tahminini Göster"))
        self.label.setText(_translate("MainWindow", "KATEGORİ "))
        self.pushButton.clicked.connect(self.clickedButtonFunction)
        self.textEdit.setPlaceholderText(_translate("MainWindow", "Haber metnini giriniz..."))
# Kategori tahminini göster butonuna basıldığında kategori tahmini yapar.
    def clickedButtonFunction(self):
        newText = list()
        newText.append(self.textEdit.toPlainText())
        newTextpreprocessingd = preprocessing(newText)
        tfidf_vector = tfidf_vectorizer.transform(newTextpreprocessingd)
        predictedClass = svm.predict(tfidf_vector)
        self.label.setText(predictedClass[0])


import sys
# Arayüzün ekrana yüklenmesi.
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()
sys.exit(app.exec_())





































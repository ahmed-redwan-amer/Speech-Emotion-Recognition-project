#librosa مكتبه تتعامل مع الصوتيات و الموسيقي و اتعرف التلقائي للكلام 
import librosa
#soundfile تستخدم لقرأءة الملفات الصوتيه و الكتايه عليها 
import soundfile
#os  هي مكتبه تستخدم وظائف انظمه التشغيل لقراءة و الكتابه علي الملفات و معالجه المساراتو  قراءة جميع الأسطر في جميع الملفات الموجودة
#glob (صوتيات مثلا) لاسترداد الملفات / المسارات التي تطابق نمطًا محددًا.
#pickleثنائية لغرض سَلسلَة وإلغاء سَلسَلَة بنية كائنات بايثون و تطلق ايضا علي على العملية التي يتحوّل فيها تسلسل هرمي لكائن بايثون إلى تدفق بايتات byte stream
import os, glob, pickle
#numpy هي مكتبة يتم استخدامها فى البايثون
#وهى مكتبة يتم استعمالها لانها تحتوى على mathimatics functions
import numpy as np
'''sklearn أدوات بسيطة وفعالة لتحليل البيانات التنبؤية
في متناول الجميع ، ويمكن إعادة استخدامها في سياقات مختلفة
مبني على NumPy و SciPy و matplotlib
و يوجد فيها كذا خاصيه Classification,Regression,Clustering,Dimensionality reduction,Model selection,Preprocessing'''
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
''' اول حاجه انا بعملها اني بعمل فانكشن عشان اسحب خواص الصوتيات الي عندي 
بيكون عندي 4 متغيرات اول حاجه الملف تاني حاجه درجه الصوت بتاعتك 
بستخرج لطبقه الصوت بتاعتك ثالث حاجه بحلل الخواص بتاعه صوتك عشانعشان اقدر احدد مشاعر صوتك عشان احدد الاموشن 
اخر حاجه اني بحدد تردد الصوت بتاعك '''
def extract_feature(file_name, mfcc, chroma, mel):
    #بستدعي المكتبه بتاعه الصوتيات عشان انفذ الاكواد الي بعد كده علي الصوت بتاعي 
    #هنحول ملفات الاصوات بتاعي الي مصفوفه 
    #هنقرئه بنوع فلوت 32
    
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        #كل فيل عندي ليه sample rate 
        #sample rate = هيرتز 
        #كل صوت بيتعمله sambling في كل ثانيه 144100
         #هنجيب متوسط mfcc
        # وهنعمل كده في chroma , mel
        sample_rate=sound_file.samplerate
        '''عشان مش كل المطلوب في الفانكشن موجود في الصوت 3 مطاليب 
          ف هنعمل شروط بحيث لو مطلوب من 3 مش موجود نقدر  نتعامل مع الصوت '''
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=48).T, axis=0)
            result=np.hstack((result, mfccs))
        
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        
        if chroma:
            stft=np.abs(librosa.stft(X))
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
       
    return result
#دي بتتجاب من الداتا مش انت الي بتالفها 
#محطوط في dic
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
#و هنا حطينا اهم الايموشن الي محتاجينها 
observed_emotions=['calm', 'happy', 'fearful', 'disgust']
#هنعمل فانكشن تحمل الداتا و تقسمها و تستخر الفبتشر بتاعتها 
#هنقسم الداتا to train and test 
#هنستخدم ربع الداتا لtest 
def load_data(test_size=0.25):
    #x= input
    #y= output 
    
    x,y=[],[]
    # ده معناه خش للمسار الي عندك ده و خش جو كل فاييل فيهم 
    #الفايل الي هتلاقي فيه كلمه Actor_ و كمان حاجه زياده 
    # يعني لو في فايل مفهوش كلمه Actor_ متدخلهوش 
    # و تخش كل الفايلات الي جوه بس شرط يكون كل الصيغ wav
    
    for file in glob.glob("D:\\PYTHON\\voice emotion recognition\\data\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        #هنستخرج الاموشن من الاصوات و هنسحبها من الفايل بتاعي 
        #و هنعمل فصل للاسم بتاعه 
        #الطبيعي انه هيفصل عن طريق المسافه 
        #بس احنا فاصلين بين الملفات عن طريق "-"
        #و تجيب العنصر رقم 2 الدال علي الايموشن 
        emotion=emotions[file_name.split("-")[2]]
        #سعات بتحصل مشاكل و يكون في ايموشن مش موجود .
        # بنقوله عديه و خش علي الاموشن الي بعده 
        if emotion not in observed_emotions:
            continue
            #هنبدا نستخرج الخواص بتاعتنا  
            
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        #بنقوله تعال في اخر X و ضيف الخواص بتاعته 
        # و ضيف في Y الايموشن 
        x.append(feature)
        y.append(emotion)
    #هبدا اعمل تدريب للداتا بتاعتي 
    x_train,x_test,y_train,y_test=train_test_split(np.array(x), y, test_size=test_size)
    return x_train,x_test,y_train,y_test
#هنبدا تحمل الداتا و يتدرب عليها 
x_train,x_test,y_train,y_test=load_data(test_size=0.25)
# ميزته الموديول ده انه بيحتوي علي nlp جواه 
#بس بيكون مبسط 
#بالنسبه ل learning_rate بتاعي بيكون adaptiev يعني انا الي بعدله بنفسي 
model=MLPClassifier(alpha=0.01, batch_size=400, epsilon=1e-09, hidden_layer_sizes=(650,), learning_rate='adaptive', max_iter=400)
#هنبدا نمرن الموديول بتاعنا 
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#معادله accuracy
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
#هنطبع accuracy بتاعه الموديول 
print("Accuracy: {:.2f}%".format(accuracy*100))

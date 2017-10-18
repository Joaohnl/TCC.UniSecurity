/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package processing;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.Iterator;
import java.util.Properties;
import java.util.Set;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.helper.opencv_core;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import org.bytedeco.javacpp.opencv_core.CvMat;
import org.bytedeco.javacpp.opencv_core.CvRect;
import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_8U;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import static org.bytedeco.javacpp.opencv_core.cvCreateImage;
import static org.bytedeco.javacpp.opencv_core.cvGetSize;
import static org.bytedeco.javacpp.opencv_core.cvLoad;
import static org.bytedeco.javacpp.opencv_core.cvSetImageROI;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvSaveImage;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.CV_INTER_LINEAR;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.cvEqualizeHist;
import static org.bytedeco.javacpp.opencv_imgproc.cvResize;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;

public class LBPHFaceRecognizer {

    private String faceDataFolder = "C:\\TCC.UniSecurity\\";
    private String imageDataFolder = faceDataFolder + "att_faces\\";
    private final String CASCADE_FILE = "C:\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
    private final String frBinary_DataFile = faceDataFolder + "frBinary.dat";
    private final String personNameMappingFileName = faceDataFolder + "personNumberMap.properties";

    private final CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(cvLoad(CASCADE_FILE));
    private Properties dataMap = new Properties();

    private final int NUM_IMAGES_PER_PERSON = 10;
    double binaryTreshold = 130;
    int highConfidenceLevel = 75;

    private FaceRecognizer fr_binary = null;

    public LBPHFaceRecognizer() {
        createModels();
        loadTrainingData();
    }

    private void createModels() {
        fr_binary = createLBPHFaceRecognizer(1, 8, 8, 8, binaryTreshold);
        //ReconizerTraining rt = new ReconizerTraining(this);
        //fr_binary.save("frBinary.dat");
    }

    public String identifyFace(IplImage image) {
        String personName = "";

        Mat imagem = new Mat(image);

        Set keys = dataMap.keySet();

        if (keys.size() > 0) {
            IntPointer ids = new IntPointer(1);
            DoublePointer distance = new DoublePointer(1);
            int result = -1;

            fr_binary.predict(imagem, ids, distance);
            //just deriving a confidence number against treshold
            result = ids.get(0);

            if (result > -1 && distance.get(0) < highConfidenceLevel) {
                personName = (String) dataMap.get("" + result) + " confiança: " + distance.get(0);
            } else {
                personName = "Não identificado!";
            }
        }

        return personName;
    }

    public String identifyFace(Mat imagem) {
        String personName = "";

        Set keys = dataMap.keySet();

        if (keys.size() > 0) {
            IntPointer ids = new IntPointer(1);
            DoublePointer distance = new DoublePointer(1);
            int result = -1;

            fr_binary.predict(imagem, ids, distance);

            //just deriving a confidence number against treshold
            result = ids.get(0);

            if (result > -1 && distance.get(0) < highConfidenceLevel) {
                personName = (String) dataMap.get("" + result) + " confiança: " + distance.get(0);
            } else {
                personName = "Não identificado!";
            }
        }

        return personName;
    }

    // The logic to learn a new face is to store the recorded images to a folder and retrain the model
    // will be replaced once update feature is available
    // Bloco esta de forma sincronizada pois as threads de identificação também podem utilizar o arquivo
    // binário para localizar a pessoa identificada
    public boolean saveNewFace(String personName, IplImage[] images) throws Exception {
        synchronized (frBinary_DataFile) {
            int memberCounter = dataMap.size();
            if (dataMap.containsValue(personName)) {
                Set keys = dataMap.keySet();
                Iterator ite = keys.iterator();
                while (ite.hasNext()) {
                    String personKeyForTraining = (String) ite.next();
                    String personNameForTraining = (String) dataMap.getProperty(personKeyForTraining);
                    if (personNameForTraining.equals(personName)) {
                        memberCounter = Integer.parseInt(personKeyForTraining);
                        System.err.println("Person already exist.. re-learning..");
                    }
                }
            }
            dataMap.put("" + memberCounter, personName);
            storeTrainingImages(personName, images);
            retrainAll();

            return true;
        }
    }

    public IplImage preprocessImage(IplImage image, CvRect r) {
        IplImage gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
        IplImage roi = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
        CvRect r1 = new CvRect(r.x() - 10, r.y() - 10, r.width() + 10, r.height() + 10);
        cvCvtColor(image, gray, CV_BGR2GRAY);
        cvSetImageROI(gray, r1);
        cvResize(gray, roi, CV_INTER_LINEAR);
        cvEqualizeHist(roi, roi);
        return roi;
    }

    public void retrainAll() throws Exception {
        Set keys = dataMap.keySet();
        if (keys.size() > 0) {
            MatVector trainImages = new MatVector(keys.size() * NUM_IMAGES_PER_PERSON);
            CvMat trainLabels = CvMat.create(keys.size() * NUM_IMAGES_PER_PERSON, 1, CV_32SC1);
            Iterator ite = keys.iterator();
            int count = 0;

            System.err.print("Loading images for training...");
            while (ite.hasNext()) {
                String personKeyForTraining = (String) ite.next();
                String personNameForTraining = (String) dataMap.getProperty(personKeyForTraining);
                IplImage[] imagesForTraining = readImages(personNameForTraining);
                IplImage grayImage = IplImage.create(imagesForTraining[0].width(), imagesForTraining[0].height(), IPL_DEPTH_8U, 1);

                for (int i = 0; i < imagesForTraining.length; i++) {
                    trainLabels.put(count, 0, Integer.parseInt(personKeyForTraining));
                    cvCvtColor(imagesForTraining[i], grayImage, CV_BGR2GRAY);
                    Mat imagemTreino = new Mat(grayImage);
                    trainImages.put(count, imagemTreino);
                    count++;
                }
                //storeNormalizedImages(personNameForTraining, imagesForTraining);
            }

            System.err.println("done.");

            System.err.print("Training Binary model ....");
            Mat imagemTreino = new Mat(trainLabels);
            fr_binary.train(trainImages, imagemTreino);
            System.err.println("done.");
            storeTrainingData();
        }

    }

    private void loadTrainingData() {

        try {
            File personNameMapFile = new File(personNameMappingFileName);
            if (personNameMapFile.exists()) {
                FileInputStream fis = new FileInputStream(personNameMappingFileName);
                dataMap.load(fis);
                fis.close();
            }

            File binaryDataFile = new File(frBinary_DataFile);
            System.err.print("Loading Binary model ....");
            fr_binary.load(frBinary_DataFile);
            System.err.println("done");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void storeTrainingData() throws Exception {
        System.err.print("Storing training models ....");

        File binaryDataFile = new File(frBinary_DataFile);
        if (binaryDataFile.exists()) {
            binaryDataFile.delete();
        }
        fr_binary.save(frBinary_DataFile);

        File personNameMapFile = new File(personNameMappingFileName);
        if (personNameMapFile.exists()) {
            personNameMapFile.delete();
        }
        FileOutputStream fos = new FileOutputStream(personNameMapFile, false);
        dataMap.store(fos, "");
        fos.close();

        System.err.println("done.");
    }

    public void storeTrainingImages(String personName, IplImage[] images) {
        for (int i = 0; i < images.length; i++) {
            String imageFileName = imageDataFolder + "training\\" + personName + "_" + i + ".bmp";
            File imgFile = new File(imageFileName);
            if (imgFile.exists()) {
                imgFile.delete();
            }
            cvSaveImage(imageFileName, images[i]);
        }
    }

    private IplImage[] readImages(String personName) {
        File imgFolder = new File(imageDataFolder);
        IplImage[] images = null;
        if (imgFolder.isDirectory() && imgFolder.exists()) {
            images = new IplImage[NUM_IMAGES_PER_PERSON];
            for (int i = 0; i < NUM_IMAGES_PER_PERSON; i++) {
                String imageFileName = imageDataFolder + "training\\" + personName + "_" + i + ".bmp";
                IplImage img = cvLoadImage(imageFileName);
                images[i] = img;
            }

        }
        return images;
    }

//    public void updateTraining(String personName, IplImage[] imagens) {
//        fr_binary.update(mv, opencv_core.AbstractMat.EMPTY);
//    }
    public int getNUM_IMAGES_PER_PERSON() {
        return NUM_IMAGES_PER_PERSON;
    }

    public String getFaceDataFolder() {
        return faceDataFolder;
    }

    public void setFaceDataFolder(String faceDataFolder) {
        this.faceDataFolder = faceDataFolder;
    }

    public String getImageDataFolder() {
        return imageDataFolder;
    }

    public void setImageDataFolder(String imageDataFolder) {
        this.imageDataFolder = imageDataFolder;
    }

    public Properties getDataMap() {
        return dataMap;
    }

    public void setDataMap(Properties dataMap) {
        this.dataMap = dataMap;
    }

    public double getBinaryTreshold() {
        return binaryTreshold;
    }

    public void setBinaryTreshold(double binaryTreshold) {
        this.binaryTreshold = binaryTreshold;
    }

//    public int getHighConfidenceLevel() {
//        return highConfidenceLevel;
//    }
//
//    public void setHighConfidenceLevel(int highConfidenceLevel) {
//        this.highConfidenceLevel = highConfidenceLevel;
//    }
    public FaceRecognizer getFr_binary() {
        return fr_binary;
    }

    public void setFr_binary(FaceRecognizer fr_binary) {
        this.fr_binary = fr_binary;
    }

}

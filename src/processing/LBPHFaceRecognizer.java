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
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import org.bytedeco.javacpp.opencv_core.CvMat;
import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_8U;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import static org.bytedeco.javacpp.opencv_core.cvLoad;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvSaveImage;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;

public class LBPHFaceRecognizer {

    private static final String faceDataFolder = "C:\\TCC.UniSecurity\\";
    private final String imageDataFolder = faceDataFolder + "src\\fotos\\";
    private final String frBinary_DataFile = faceDataFolder + "src\\recursos\\frBinary.dat";
    private static final String personNameMappingFileName = faceDataFolder + "src\\recursos\\personNumberMap.properties";

    private Properties dataMap = new Properties();

    private ArduinoSerial portaCOM;

    private final int NUM_IMAGES_PER_PERSON = 10;
    double binaryTreshold = 130;
    int highConfidenceLevel = 75;

    private FaceRecognizer fr_binary = null;

    public LBPHFaceRecognizer() {
        portaCOM = new ArduinoSerial("COM3");
        portaCOM.initialize();
        createModels();
        loadTrainingData();

    }

    private void createModels() {
        fr_binary = createLBPHFaceRecognizer(1, 8, 8, 8, binaryTreshold);
//        ReconizerTraining rt = new ReconizerTraining(this);
//        fr_binary.save("frBinary.dat");
    }

    public String identifyFace(IplImage image) {
        synchronized (dataMap) {
            String personName = "";

            Mat imagem = new Mat(image);

            Set keys = dataMap.keySet();

            if (keys.size() > 0) {
                IntPointer ids = new IntPointer(1);
                DoublePointer distance = new DoublePointer(1);
                int result = -1;

                fr_binary.predict(imagem, ids, distance);
                // Derivando nível de confiança contra o threshold
                result = ids.get(0);

                if (result > -1 && distance.get(0) < highConfidenceLevel) {
                    personName = (String) dataMap.get("" + result) + " confiança: " + distance.get(0);
                    if (personName != null) {
                        portaCOM.send("l");
                        portaCOM.send("d");
                    }
                } else {
                    personName = "Não identificado!";
                }
            }
            return personName;
        }
    }

    public String identifyFace(Mat imagem) {
        synchronized (dataMap) {
            String personName = "";

            Set keys = dataMap.keySet();

            if (keys.size() > 0) {
                IntPointer rotulo = new IntPointer(1);
                DoublePointer confianca = new DoublePointer(1);
                int result = -1;

                fr_binary.predict(imagem, rotulo, confianca);
                // Derivando nível de confiança contra o threshold
                result = rotulo.get(0);

                if (result > -1 && rotulo.get(0) < highConfidenceLevel) {
                    personName = (String) dataMap.get("" + result) + " confianca: " + confianca.get(0);
                    if (personName != null) {
                        portaCOM.send("l");
                        portaCOM.send("d");
                    }
                } else {
                    personName = "Não identificado!";
                }
            }

            return personName;
        }
    }

    // The logic to learn a new face is to store the recorded images to a folder and retrain the model
    // will be replaced once update feature is available
    // Bloco esta de forma sincronizada pois as threads de identificação também podem utilizar o arquivo
    // binário para localizar a pessoa identificada
    public boolean saveNewFace(String personName, IplImage[] images) throws Exception {
        int memberCounter = dataMap.size();
        if (dataMap.containsValue(personName)) {
            Set keys = dataMap.keySet();
            Iterator ite = keys.iterator();
            while (ite.hasNext()) {
                String personKeyForTraining = (String) ite.next();
                String personNameForTraining = (String) dataMap.getProperty(personKeyForTraining);
                if (personNameForTraining.equals(personName)) {
                    memberCounter = Integer.parseInt(personKeyForTraining);
                    System.err.println("Pessoa já existe no banco de dados.. aprendendo novamente..");
                }
            }
        }
        dataMap.put("" + memberCounter, personName);
        storeTrainingImages(personName, images);
        retrainAll();

        return true;

    }

    public void retrainAll() throws Exception {
        synchronized (fr_binary) {
            synchronized (dataMap) {
                synchronized (frBinary_DataFile) {

                    Set keys = dataMap.keySet();
                    if (keys.size() > 0) {
                        MatVector trainImages = new MatVector(keys.size() * NUM_IMAGES_PER_PERSON);
                        CvMat trainLabels = CvMat.create(keys.size() * NUM_IMAGES_PER_PERSON, 1, CV_32SC1);
                        Iterator ite = keys.iterator();
                        int count = 0;

                        System.err.print("Carregando imagens para treinamento...");
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

                        System.err.println("concluído.");

                        System.err.print("Treinando modelo binário ....");
                        Mat imagemTreino = new Mat(trainLabels);
                        fr_binary.train(trainImages, imagemTreino);
                        System.err.println("concluído.");
                        storeTrainingData();
                    }
                }
            }
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

            System.err.print("Carregando modelo binário ....");
            fr_binary.load(frBinary_DataFile);
            System.err.println("concluído.");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void storeTrainingData() throws Exception {
        synchronized (fr_binary) {
            synchronized (dataMap) {

                System.err.print("Salvando modelo binário ....");

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

                System.err.println("concluído.");
            }
        }
    }

    public void storeTrainingImages(String personName, IplImage[] images) {
        synchronized (imageDataFolder) {
            for (int i = 0; i < images.length; i++) {
                String imageFileName = imageDataFolder + personName + "-" + i + ".jpg";
                File imgFile = new File(imageFileName);
                if (imgFile.exists()) {
                    imgFile.delete();
                }
                cvSaveImage(imageFileName, images[i]);
            }
        }
    }

    private IplImage[] readImages(String personName) {
        synchronized (imageDataFolder) {
            File imgFolder = new File(imageDataFolder);
            IplImage[] images = null;
            if (imgFolder.isDirectory() && imgFolder.exists()) {
                images = new IplImage[NUM_IMAGES_PER_PERSON];
                for (int i = 1; i <= NUM_IMAGES_PER_PERSON; i++) {
                    String imageFileName = imageDataFolder + personName + "-" + i + ".jpg";
                    IplImage img = cvLoadImage(imageFileName);
                    images[i - 1] = img;
                    if (images[i - 1] == null) {
                        continue;
                    }

                }
            }
            return images;
        }

    }

    public void updateTraining(String personName, IplImage[] imagens) {
        // fr_binary.update(mv,imagens);
    }

    public int getNUM_IMAGES_PER_PERSON() {
        return NUM_IMAGES_PER_PERSON;
    }

    public Properties getDataMap() {
        return dataMap;
    }

}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package processing;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.Timer;
import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_PLAIN;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;

import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_objdetect.CASCADE_FIND_BIGGEST_OBJECT;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;

public class PainelReconhecimento implements Runnable {

    // Modelo cascata utilizado para detecção das faces (Disponibilizado pelo OpenCV)
    private static final String FACE_CASCADE = "haarcascade_frontalface_alt.xml";
    private CascadeClassifier detectorFace;

    private boolean executando = true;

    private final int FACE_LARGURA = 160;
    private final int FACE_ALTURA = 160;

    private LBPHFaceRecognizer lbphRecognizer;

    private CanvasFrame cFrame;
    private Frame frameCapturado;
    private Mat imagemColorida;

    private OpenCVFrameConverter.ToMat converteMat;
    private OpenCVFrameGrabber camera;

    public PainelReconhecimento(LBPHFaceRecognizer lbphRecognizer) throws FrameGrabber.Exception {
        converteMat = new OpenCVFrameConverter.ToMat();
        camera = new OpenCVFrameGrabber(0);
        camera.setImageWidth(700);
        camera.setImageHeight(540);
        this.lbphRecognizer = lbphRecognizer;

        detectorFace = new CascadeClassifier(FACE_CASCADE);

        cFrame = new CanvasFrame("Reconhecimento", CanvasFrame.getDefaultGamma() / camera.getGamma());
        cFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        cFrame.setLocation(621, 0);
        frameCapturado = null;
        imagemColorida = new Mat();

        //reconheceFace();
    }

    private void reconheceFace() throws FrameGrabber.Exception {
        // Inicia câmera
        camera.start();
        while ((frameCapturado = camera.grab()) != null && executando) {
            imagemColorida = converteMat.convert(frameCapturado);

            //Converte a imagem da câmera para escalas de cinza
            Mat imagemCinza = new Mat();
            cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);

            //Realiza a detecção da maior face e armazena seus dados em um vetor facesDetectadas
            RectVector facesDetectadas = new RectVector();
            detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 2,
                    CASCADE_FIND_BIGGEST_OBJECT, new Size(100, 100), new Size(500, 500));

            int totalFaces = (int) facesDetectadas.size();
            if (totalFaces == 0) {
                // Caso não encontre nenhuma face, informa na tela
                putText(imagemColorida, "Nenhuma face encontrada!", new Point(20, 20), FONT_HERSHEY_PLAIN, 1.4, new Scalar(0, 0, 255, 0));
            } else {
                Rect dadosFace = facesDetectadas.get(0);
                //Desenha o retângulo ao redor da face detectada
                rectangle(imagemColorida, dadosFace, new Scalar(0, 255, 0, 0));

                // Extrai a imagem para um Mat e redimensiona
                Mat faceCapturada = new Mat(imagemCinza, dadosFace);
                resize(faceCapturada, faceCapturada, new Size(FACE_LARGURA, FACE_ALTURA));

                // Chamada de método para reconhecimento da face extraída
                String nome = lbphRecognizer.identificaFace(faceCapturada);

                //Informa o reconhecimento
                putText(imagemColorida, nome, new Point(20, 20), FONT_HERSHEY_PLAIN, 1.2, new Scalar(0, 0, 255, 0));

            }
            if (cFrame.isVisible()) {
                // Exibe imagem extraída da câmera no frame
                cFrame.showImage(frameCapturado);
            }
        }
        //Encerra e câmera e fecha o frame de captura
        cFrame.dispose();
        camera.stop();
    }

    public void PararExecucao() {
        this.executando = false;
    }

    @Override
    public void run() {
        try {
            // Inicia câmera
            camera.start();
            while ((frameCapturado = camera.grab()) != null && executando) {
                imagemColorida = converteMat.convert(frameCapturado);
                
                //Converte a imagem da câmera para escalas de cinza
                Mat imagemCinza = new Mat();
                cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);
                
                //Realiza a detecção da maior face e armazena seus dados em um vetor facesDetectadas
                RectVector facesDetectadas = new RectVector();
                detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 2,
                        CASCADE_FIND_BIGGEST_OBJECT, new Size(100, 100), new Size(500, 500));
                
                int totalFaces = (int) facesDetectadas.size();
                if (totalFaces == 0) {
                    // Caso não encontre nenhuma face, informa na tela
                    putText(imagemColorida, "Nenhuma face encontrada!", new Point(20, 20), FONT_HERSHEY_PLAIN, 1.4, new Scalar(0, 0, 255, 0));
                } else {
                    Rect dadosFace = facesDetectadas.get(0);
                    //Desenha o retângulo ao redor da face detectada
                    rectangle(imagemColorida, dadosFace, new Scalar(0, 255, 0, 0));
                    
                    // Extrai a imagem para um Mat e redimensiona
                    Mat faceCapturada = new Mat(imagemCinza, dadosFace);
                    resize(faceCapturada, faceCapturada, new Size(FACE_LARGURA, FACE_ALTURA));
                    
                    // Chamada de método para reconhecimento da face extraída
                    String nome = lbphRecognizer.identificaFace(faceCapturada);
                    
                    //Informa o reconhecimento
                    putText(imagemColorida, nome, new Point(20, 20), FONT_HERSHEY_PLAIN, 1.2, new Scalar(0, 0, 255, 0));
                    
                }
                if (cFrame.isVisible()) {
                    // Exibe imagem extraída da câmera no frame
                    cFrame.showImage(frameCapturado);
                }
            }
            //Encerra e câmera e fecha o frame de captura
            cFrame.dispose();
            camera.stop();
            camera.release();
        } catch (FrameGrabber.Exception ex) {
            Logger.getLogger(PainelReconhecimento.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

}

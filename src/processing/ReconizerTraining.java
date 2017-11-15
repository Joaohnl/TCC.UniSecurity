/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package processing;

import java.io.File;
import java.io.FilenameFilter;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.bytedeco.javacpp.opencv_core.IplImage;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;

/**
 *
 * @author Joao_
 */
public class ReconizerTraining {

    private static String caminhoBD = "C:\\TCC.UniSecurity\\src\\fotos\\";
    private static LBPHFaceRecognizer lbph;

    public static void main(String args[]) {

        lbph = new LBPHFaceRecognizer();
        
        FilenameFilter imgFilter = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
            }
        };

        File pastaImagens;
        File[] arquivoImagens;
        IplImage[][] imagens;
        int ID, nrFoto;
        String[] identificacoes;

        pastaImagens = new File(caminhoBD);
        arquivoImagens = pastaImagens.listFiles(imgFilter);

        identificacoes = new String[arquivoImagens.length / 10];
        imagens = new IplImage[arquivoImagens.length / 10][10];
        ID = 0;
        for (File imagem : arquivoImagens) {
            IplImage img = cvLoadImage(imagem.getAbsolutePath(), CV_BGR2GRAY);
            nrFoto = Integer.parseInt(imagem.getName().substring(imagem.getName().indexOf('-') + 1, imagem.getName().indexOf('.')));
            identificacoes[ID] = imagem.getName().substring(0, imagem.getName().indexOf('-'));

            
            
            System.out.println(ID);
            System.out.println(nrFoto);
            System.out.println(identificacoes[ID]);
            
            
            imagens[ID][nrFoto - 1] = img;
        }
        try {
            for (int i = 0; i < imagens.length; i++) {

                lbph.saveNewFace(identificacoes[i], imagens[i]);
            }

            lbph.retrainAll();
        } catch (Exception ex) {
            Logger.getLogger(ReconizerTraining.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}

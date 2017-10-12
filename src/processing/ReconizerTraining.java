/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package processing;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.IntBuffer;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 *
 * @author Joao_
 */
public class ReconizerTraining {

    private String caminhoBD = "D:\\TCC\\br.unisantos.UniSecurity2\\att_faces\\";

    public ReconizerTraining(FaceRecognizer facerecognizer) {

        FilenameFilter imgFilter = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
            }
        };

        try (FileReader arq = new FileReader(caminhoBD + "attfaces.txt")) {
            String caminhoPastaImagens;
            int identificacao;
            File pastaImagens;
            File[] arquivoImagens;
            MatVector imagens;
            Mat identificacoes;
            IntBuffer bufferIdentificacoes;
            int counter;

            BufferedReader lerArq = new BufferedReader(arq);
            String linha = lerArq.readLine();
            while (linha != null) {
                caminhoPastaImagens = linha.substring(0, linha.indexOf(";"));
                identificacao = Integer.parseInt(linha.substring(linha.indexOf(";") + 1, linha.length()));

                pastaImagens = new File(caminhoPastaImagens);
                arquivoImagens = pastaImagens.listFiles(imgFilter);

                imagens = new MatVector(arquivoImagens.length);

                identificacoes = new Mat(arquivoImagens.length, 1, CV_32SC1);
                bufferIdentificacoes = identificacoes.createBuffer();

                counter = 0;

                for (File imagem : arquivoImagens) {
                    Mat img = imread(imagem.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);

                    imagens.put(counter, img);
                    
                    bufferIdentificacoes.put(counter, identificacao);
                    
                    counter++;
                }
                facerecognizer.train(imagens, identificacoes);
                linha = lerArq.readLine();
            }

            arq.close();
        } catch (IOException e) {
            System.err.printf("Erro na abertura do arquivo: %s.\n",
                    e.getMessage());
        }
    }
}

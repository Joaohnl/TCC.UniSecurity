/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package processing;

import gui.telaPrincipal;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.IntBuffer;
import java.util.Properties;
import java.util.Set;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class LBPHFaceRecognizer {

    private static final String caminhoMapaDados = "src\\recursos\\personNumberMap.properties";
    private final String pastaImagens = "src\\fotos\\";
    private final String LBPH_DADOS = "src\\recursos\\frBinary.dat";

    private Properties mapaDados = new Properties();

    private ArduinoSerial portaCOM;

    private final int NUMERO_IMAGENS_PESSOA = 10;
    double Threshold = 130;
    int nivelConfianca = 75;

    private FaceRecognizer lbph = null;

    public LBPHFaceRecognizer() {
        portaCOM = new ArduinoSerial("COM3");
        portaCOM.initialize();
        criaModelo();
        carregaDados();
    }

    private void criaModelo() {
        lbph = createLBPHFaceRecognizer(1, 8, 8, 8, Threshold);
    }
    
    public String identificaFace(Mat imagem) {
        String nomePessoa = "";

        // Carrega as chaves contidas no mapaDados
        Set chaves = mapaDados.keySet();

        if (chaves.size() > 0) {
            // Variáveis para identificação
            IntPointer rotulo = new IntPointer(1);
            DoublePointer confianca = new DoublePointer(1);
            int resultado = -1;

            // Chama o método de reconhecimento
            lbph.predict(imagem, rotulo, confianca);
            // Derivando nível de confiança contra o threshold
            resultado = rotulo.get(0);

            // Verifica se a identificação é confiável
            if (resultado > -1 && confianca.get(0) < nivelConfianca) {
                //Se sim, atribui à variável nomePessoa o respectivo nome extraído do mapaDados
                nomePessoa = (String) mapaDados.get("" + resultado) + " confianca: " + confianca.get(0);
                
                // Emite o sinal para a liberação da catraca
                portaCOM.send("l");
                portaCOM.send("d");
                
            } else {
                // Caso identificação não seja confiável, atribui à variável nomePessoa como "Não identificada"
                nomePessoa = "Não identificado!";
            }
        }
        // Retorna nome da pessoa identificada, ou "Não identificado" caso a identificação não for bem sucedida
        return nomePessoa;
    }

    private void carregaDados() {

        try {
            File arquivoMapaDados = new File(caminhoMapaDados);
            if (arquivoMapaDados.exists()) {
                FileInputStream fis = new FileInputStream(caminhoMapaDados);
                mapaDados.load(fis);
                fis.close();
            }

            telaPrincipal.SetTextoLog("Carregando modelo binário ....");
            lbph.load(LBPH_DADOS);
            telaPrincipal.SetTextoLog("concluído.\n");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void salvaDados() throws Exception {

        telaPrincipal.SetTextoLog("Salvando modelo binário ....");

        File arquivoBinario = new File(LBPH_DADOS);
        if (arquivoBinario.exists()) {
            arquivoBinario.delete();
        }
        lbph.save(LBPH_DADOS);

        File arquivoMapaDados = new File(caminhoMapaDados);
        if (arquivoMapaDados.exists()) {
            arquivoMapaDados.delete();
        }
        FileOutputStream fos = new FileOutputStream(arquivoMapaDados, false);
        mapaDados.store(fos, "");
        fos.close();

        telaPrincipal.SetTextoLog("concluído.\n");
    }

    public int Treinamento() {
        try {
            File diretorio = new File(pastaImagens);
            FilenameFilter filtroImagem = (File dir, String nome) -> 
                    {
                return nome.endsWith(".jpg") || nome.endsWith(".gif") || nome.endsWith(".png") || nome.endsWith(".pgm");
            };
            
            telaPrincipal.SetTextoLog("Carregando imagens para treinamento...\n");
            File[] arquivos = diretorio.listFiles(filtroImagem);
            if (arquivos.length == 0) {
                mapaDados.clear();
                lbph.clear();
                salvaDados();
                return 1;
            }
            opencv_core.MatVector fotos = new opencv_core.MatVector(arquivos.length);
            opencv_core.Mat rotulos = new opencv_core.Mat(arquivos.length, 1, CV_32SC1);
            IntBuffer rotulosBuffer = rotulos.createBuffer();
            int contador = 0;
            for (File imagem : arquivos) {
                opencv_core.Mat foto = imread(imagem.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
                int classe = Integer.parseInt(imagem.getName().split("\\.")[0]);
                String nome = imagem.getName().split("\\.")[1];

                //System.out.println(classe);   //Imprime a classe de cada grupo de imagem (para debug)
                resize(foto, foto, new opencv_core.Size(160, 160));
                fotos.put(contador, foto);
                rotulosBuffer.put(contador, classe);
                if (!mapaDados.contains(nome)) {
                    mapaDados.put("" + classe, nome);
                }
                contador++;
            }
            telaPrincipal.SetTextoLog("Treinando modelo binário...");
            lbph.train(fotos, rotulos);
            telaPrincipal.SetTextoLog("concluído!\n");
            salvaDados();
            return 0;
        } catch (Exception ex) {
            return 2;
        }
    }

    public int getNUMERO_IMAGENS_PESSOA() {
        return NUMERO_IMAGENS_PESSOA;
    }

    public Properties getDataMap() {
        return mapaDados;
    }

}

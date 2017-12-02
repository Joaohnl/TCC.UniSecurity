package processing;

/* 
    Este painel repetidamente tira imagens do dispositivo de captura e as desenha no painel.
    Uma face é destacada com um retângulo verde, que é atualizado conforme a face se movimenta.
    
    A face destacada pode ser armazenada para reconhecimento da pessoa.

    A tarefa de detecção é realizado pelo Haar cascade classificador disponibilizado pelo JavaCV.
    Esta tarefa é executada em sua própria thread devido o processo ser mais lento. sendo assim
    a captura das imagens não é afetada pelo reconhecimento dos rostos.
 */

import gui.telaPrincipal;
import java.awt.*;
import javax.swing.*;
import java.awt.image.*;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import javax.imageio.ImageIO;
import org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacpp.opencv_core.Point;

public class PainelCadastro extends JPanel implements Runnable {

    /* Dimensões de cada imagem; o painel é do mesmo tamanho das imagens */
    private static final int LARGURA = 720;
    private static final int ALTURA = 560;

    private static final int DELAY = 100;  // Tempo (ms) delay para desenhar o painel

    private static final int CAMERA_ID = 0;

    private static final int ESCALA_IMAGEM = 4;

    // tempo (ms) entre cada detecçãoo de face
    private static final int DELAY_DETECCAO = 500;

    private static final int MAX_TAREFAS = 4;
    // Nr. máximo de tarefas que podem ficar aguardando no executor

    // Modelo cascata utilizado para detecção das faces (Disponibilizado pelo OpenCV)
    private static final String FACE_CASCADE = "haarcascade_frontalface_alt.xml";
    // "haarcascade_frontalface_alt2.xml";

    // Atributos para salvar uma imagem
    private static final String DIR_FACE = "src//fotos";
    private static String FACE_NOME;
    private static int FACE_ID;
    private static int AMOSTRA = 0;
    private static final int FACE_LARGURA = 160;
    private static final int FACE_ALTURA = 160;
    private static volatile boolean salvarFace = false;
    
    // Armazena as coordenadas da face
    private Rectangle faceRect;             
    

    private IplImage snapIm = null;
    private volatile boolean estaRodando;
    private volatile boolean estaTerminado;

    // Usado para disponibilizar tempo da captura das imagens
    private int contadorImagem = 0;
    private long tempoTotal = 0;
    private Font msgFonte;

    // Variáveis do JavaCV
    private FrameGrabber captura = null;
    private CvHaarClassifierCascade classificador;
    private CvMemStorage armazenamento;
    private IplImage imagemCinza;

    // Usado para as threads que executam a detecção das faces
    private ExecutorService executor;
    private AtomicInteger numTarefas;

    // Usado para armazenar número de tarefas de detecção
    private long inicioDeteccao = 0;
    
    private OpenCVFrameConverter.ToIplImage conversor;

    public PainelCadastro() {
        setBackground(Color.white);
        msgFonte = new Font("SansSerif", Font.BOLD, 14);

        executor = Executors.newSingleThreadExecutor();
        /* O executor controla uma Ãºnica thread com uma fila.
            Somente uma tarefa pode executar de cada vez. As outras devem esperar.
         */
        numTarefas = new AtomicInteger(0);
        // Usado para limitar o tamanho da fila do executor.

        criaDetector();
        faceRect = new Rectangle();

        estaRodando = true;
        estaTerminado = false;

        new Thread(this).start();   // Atualiza as imagens para o painel.
    }
    

    private void criaDetector() {
        // Cria uma instância do cascade classificador para detecção dos rostos
        classificador = new CvHaarClassifierCascade(cvLoad(FACE_CASCADE));
        if (classificador.isNull()) {
            telaPrincipal.SetTextoLog("\nNão foi possível carregar o cascade: " + FACE_CASCADE);
            System.exit(1);
        }

        armazenamento = CvMemStorage.create();  //Cria armazenamento usado para detecção

    } // Fim de initDetector()

    /*
        Deixa o painel no tamanho suficiente para uma imagem
     */
    @Override
    public Dimension getPreferredSize() {
        return new Dimension(LARGURA, ALTURA);
    }
    
    /*
        Exibe a imagem do dispositivo de captura a cada tempo de DELAY.
        A tarefa de detecção somente se inicia após cada tempo de DELAY_DETECCAO
        e somente se o número de tarefas no executor for menor que seu limite
     */
    @Override
    public void run() {
        captura = iniciaCaptura(CAMERA_ID);
        if (captura == null) {
            return;
        }

        long duracao;

        while (estaRodando) {
            long tempoInicio = System.currentTimeMillis();

            snapIm = capturaImagem(captura, CAMERA_ID);

            if (((System.currentTimeMillis() - inicioDeteccao) > DELAY_DETECCAO)
                    && (numTarefas.get() < MAX_TAREFAS)) {
                buscaFace(snapIm);
            }
            contadorImagem++;
            repaint();

            duracao = System.currentTimeMillis() - tempoInicio;
            tempoTotal += duracao;

            if (duracao < DELAY) {
                try {
                    Thread.sleep(DELAY - duracao);  // Aguarda terminar o tempo de delay
                } catch (InterruptedException ex) {
                    ex.printStackTrace();
                }
            }

        }
        encerraCaptura(CAMERA_ID);
        estaTerminado = true;
    }
    

    /*
        Inicia objeto de captura
     */
    private FrameGrabber iniciaCaptura(int ID) {
        telaPrincipal.SetTextoLog("Inicializando captura pelo dispositivo: " + ID + "\n");
        //telaPrincipal.SetTextoLog("Inicializando captura pelo dispositivo: " + videoInput.getDeviceName(ID));
        try {
            captura = FrameGrabber.createDefault(ID);
            captura.setFormat("dshow");       // Usando DirectShow
            captura.setImageWidth(LARGURA);     // tamanho padrão das imagens é pequeno: 320x240
            captura.setImageHeight(ALTURA);

            captura.start();
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }
        return captura;
    }
    
    
    private FrameGrabber iniciaCaptura(String ID) {
        telaPrincipal.SetTextoLog("Inicializando captura pelo dispositivo: " + ID + "\n");
        //telaPrincipal.SetTextoLog("Inicializando captura pelo dispositivo: " + videoInput.getDeviceName(ID));
        try {
            captura = new FFmpegFrameGrabber(ID);
            //captura.setFormat("dshow");       // Usando DirectShow
            captura.setImageWidth(LARGURA);     // tamanho padrÃ£o das imagens Ã© pequeno: 320x240
            captura.setImageHeight(ALTURA);

            captura.start();
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }
        return captura;
    }
    

    private IplImage capturaImagem(FrameGrabber captura, int ID) {
        IplImage imagem = null;
        conversor = new OpenCVFrameConverter.ToIplImage();
        try {
            imagem = conversor.convert(captura.grab());  //Tira uma foto
        } catch (FrameGrabber.Exception e) {
            telaPrincipal.SetTextoLog("Problema ao carregar imagem da câmera " + ID);
        }
        return imagem;
    }
    
    
    private IplImage capturaImagem(FrameGrabber captura, String ID) {
        IplImage imagem = null;
        conversor = new OpenCVFrameConverter.ToIplImage();
        try {
            imagem = conversor.convert(captura.grab());  //Tira uma foto
        } catch (FrameGrabber.Exception e) {
            telaPrincipal.SetTextoLog("Problema ao carregar imagem da câmera " + ID);
        }
        return imagem;
    }
    

    /*
        Finaliza a obtenção das imagens do dispositivo de cÃ¢mera
     */
    public void encerraCaptura(int ID) {
        try {
            captura.stop();
            captura.release();
        } catch (FrameGrabber.Exception e) {
            telaPrincipal.SetTextoLog("Problema ao desativar captura da câmera " + ID);
        }
    }
    
    
    public void encerraCaptura(String ID) {
        try {
            captura.stop();
            captura.release();
        } catch (FrameGrabber.Exception e) {
            telaPrincipal.SetTextoLog("Problema ao desativar captura da câmera " + ID);
        }
    }
    

    /*
        Desenha a imagem, o retângulo em volta a da face detectada e a média de tempo
        de obtenção das imagens da cÃ¢mera no canto inferior esquerdo do painel.
        No canto superior exibe a face identificada.
        O tempo exibido não inclui a tarefa de detecçãoo do rosto.
     */
    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setFont(msgFonte);

        // Desenha a imagem, estatísticas e retângulo
        if (snapIm != null) {
            g2.setColor(Color.YELLOW);
            OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
            Java2DFrameConverter paintConverter = new Java2DFrameConverter();
            org.bytedeco.javacv.Frame frame = grabberConverter.convert(snapIm);
            g2.drawImage(paintConverter.getBufferedImage(frame, 1), 0, 0, this);   // Desenha a imagem
            
            desenhaRetangulo(g2);
        } else {  // Caso ainda não obtiver nenhuma imagem
            g2.setColor(Color.BLUE);
            g2.drawString("Carregando câmera: " + CAMERA_ID, 5, ALTURA - 10);
        }
    }
    

    /*
        Usa o retângulo da face a desenhar o retÃ¢ngulo verde em torno da face
        O desenho do retângulo está em um bloco sincronizado pois a variável faceRect pode estar sendo
        utilizada em outra thread.
     */
    private void desenhaRetangulo(Graphics2D g2) {
        synchronized (faceRect) {
            if (faceRect.width == 0) {
                return;
            }

            // Desenha o retângulo
            g2.setColor(Color.GREEN);
            g2.setStroke(new BasicStroke(2));
            g2.drawRect(faceRect.x, faceRect.y, faceRect.width, faceRect.height);

        }
    }
    

    /*  Termina a execução do run() e espera a finalização do sistema.
        Para a aplicação até que sua execução termine.
     */
    public void closeDown() {
        estaRodando = false;
        while (!estaTerminado) {
            try {
                Thread.sleep(DELAY);
            } catch (InterruptedException ex) {
            }
        }
    }
    

    // ------------------------- Detecção da Face ----------------------------\\
    /*  Cria um thread para detectar faces nas imagens geradas pela câmera.
        Armazena as coordenadas das faces em faceRect e salva a imagem para treinamento caso solicitado.
        Imprime o tempo de execução no console
     */
    private void buscaFace(IplImage img) {
        numTarefas.getAndIncrement();     // Incrementa o número de tarefas antes de entrar na fila
        executor.execute(() -> {
            inicioDeteccao = System.currentTimeMillis();
            CvRect rect = detectaFace(img);
            if (rect != null) {
                // Seta parâmetros do retângulo
                setRectangle(rect);
            } else {
                faceRect.setBounds(0, 0, 0, 0);
            }
            long duracaoDeteccao = System.currentTimeMillis() - inicioDeteccao;
            //telaPrincipal.SetTextoLog(" detection duration: " + detectDuration + "ms");
            numTarefas.getAndDecrement();  // Decrementa o nÃºmero de tarefas quando terminado
        });
    }
    

    /*  Formata a imagem e converte para escala de cinza. Formatação deixa a imagem menor
        tornando o processamento mais rápido.
        O detector Haar precisa como parâmetro a imagem em escala de cinza    
     */
 /*  Detecta somente uma face utilizando o Haar Detector
     */
    private CvRect detectaFace(IplImage imagemColorida) {

        // Converte para escalas de Cinza
        imagemCinza = escalaCinza(imagemColorida);

        CvSeq faces = cvHaarDetectObjects(imagemCinza, classificador, armazenamento, 1.1, 1, // 3
                // CV_HAAR_SCALE_IMAGE |
                CV_HAAR_DO_ROUGH_SEARCH | CV_HAAR_FIND_BIGGEST_OBJECT);
        // Por questões de segurança e rapidez, só será detectado o rosto maior e mais próximo.

        int total = faces.total();
        if (total == 0) {
            telaPrincipal.SetTextoLog("Face não encontrada\n");
            //telaPrincipal.SetTextoLog("No faces found");
            return null;
        } else if (total > 1) //Este caso não deveria ocorrer. Incluído por segurança
        {
            telaPrincipal.SetTextoLog("Multiplas faces detectadas (" + total + "); utilizando a primeira\n");
            // telaPrincipal.SetTextoLog("Multiple faces detected (" + total + "); using the first");
        } else {
            telaPrincipal.SetTextoLog("Face detectada\n");
            //telaPrincipal.SetTextoLog("Face detected");
        }

        CvRect rect = new CvRect(cvGetSeqElem(faces, 0));

        if (salvarFace) {
            CortarImagemSalva(imagemColorida);
        }

        cvClearMemStorage(armazenamento);
        return rect;
    }

    /* Extrai o tamanho e as coordenadas da imagem desatacada da estrutura do retângulo do JavaCV
       e armazena em um retângulo Java.
       Durante o processo, desfaz o escalonamento que foi aplicado a imagem antes da detecção do rosto.
       O uso dessa função está em um bloco sincronizado pois o retângulo pode estar sendo utilizado para
       atualizar o painel de imagem ou pintura do retângulo ao mesmo tempo em outras threads.
     */
    private void setRectangle(CvRect r) {
        synchronized (faceRect) {
            int novoX = r.x() * ESCALA_IMAGEM;
            int novoY = r.y() * ESCALA_IMAGEM;
            int novaLargura = r.width() * ESCALA_IMAGEM;
            int novaAltura = r.height() * ESCALA_IMAGEM;

            faceRect.setRect(novoX, novoY, novaLargura, novaAltura);
        }
    }

    // --------------------------- Salvar e Aprender nova Face -----------------------------------
    public static void SalvarFace(String faceNome, int faceID, int amostra) {
        salvarFace = true;
        FACE_NOME = faceNome;
        FACE_ID = faceID;
        AMOSTRA = amostra;
    }

    private void CortarImagemSalva(IplImage img) /* clip the image using the current face rectangle, and save it into fnm
     The use of faceRect is in a synchronized block since it may be being
     updated or used for drawing at the same time in other threads.
     */ {
        BufferedImage imagemCortada = null;
        synchronized (faceRect) {
            if (faceRect.width == 0) {
                telaPrincipal.SetTextoLog("Nenhuma face selecionada!\n");
                return;
            }
            BufferedImage imagem = IplImageToBufferedImage(img);
            try {
                imagemCortada = imagem.getSubimage(faceRect.x, faceRect.y, faceRect.width, faceRect.height);
            } catch (RasterFormatException e) {
                telaPrincipal.SetTextoLog("Não foi possível cortar a imagem!\n");
            }
        }
        if (imagemCortada != null) {
            SalvarCorte(imagemCortada);
        }
    }  // end of CortarImagemSalva()

    private void SalvarCorte(BufferedImage imagemCortada) /* resizes to at least a standard size, converts to grayscale, 
     clips to an exact size, then saves in a standard location */ {
        long tempoInicio = System.currentTimeMillis();

        telaPrincipal.SetTextoLog("Salvando imagem...\n");
        BufferedImage imagemCinza = RedimensionarImagem(imagemCortada);
        BufferedImage imagemFace = CortarFace(imagemCinza);
        SalvarImagem(imagemFace, DIR_FACE + "/" + FACE_ID + "." + FACE_NOME + "." + AMOSTRA + ".jpg");
        AMOSTRA++;

        salvarFace = false;
        telaPrincipal.SetTextoLog("  Tempo de armazenamento: " + (System.currentTimeMillis() - tempoInicio) + " ms\n");
    }  // end of SalvarCorte()

    private BufferedImage RedimensionarImagem(BufferedImage im) // resize to at least a standard size, then convert to grayscale 
    {
        // resize the image so *at least* FACE_LARGURA*FACE_ALTURA size
        int imLargura = im.getWidth();
        int imAltura = im.getHeight();
        telaPrincipal.SetTextoLog("Original (w,h): (" + imLargura + ", " + imAltura + ")\n");

        double escalaLargura = FACE_LARGURA / ((double) imLargura);
        double escalaAltura = FACE_ALTURA / ((double) imAltura);
        double escala = (escalaLargura > escalaAltura) ? escalaLargura : escalaAltura;

        int novaLargura = (int) Math.round(imLargura * escala);
        int novaAltura = (int) Math.round(imAltura * escala);

        Graphics2D g2;
        if (AMOSTRA == 1) {
            BufferedImage colorida = new BufferedImage(novaLargura, novaAltura, COLOR_BGR2RGB);
            g2 = colorida.createGraphics();
            g2.drawImage(im, 0, 0, novaLargura, novaAltura, 0, 0, imLargura, imAltura, null);
            SalvarImagem(colorida, DIR_FACE + "/" + FACE_ID + "." + FACE_NOME + "." + (AMOSTRA - 1) + ".jpg");
        }
        
        // convert to grayscale while resizing
        BufferedImage imagemCinza = new BufferedImage(novaLargura, novaAltura,
                BufferedImage.TYPE_BYTE_GRAY);
        g2 = imagemCinza.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2.drawImage(im, 0, 0, novaLargura, novaAltura, 0, 0, imLargura, imAltura, null);
        g2.dispose();

        telaPrincipal.SetTextoLog("Escala de cinza (L,A): (" + novaLargura + ", " + novaAltura + ")\n");
        return imagemCinza;
    }  // end of RedimensionarImagem()

    private BufferedImage CortarFace(BufferedImage im) // clip image to FACE_LARGURA*FACE_ALTURA size
    // I assume the input image is face size or bigger
    {
        int xOffset = (im.getWidth() - FACE_LARGURA) / 2;
        int yOffset = (im.getHeight() - FACE_ALTURA) / 2;
        BufferedImage imagemFace = null;
        try {
            imagemFace = im.getSubimage(xOffset, yOffset, FACE_LARGURA, FACE_ALTURA);
            telaPrincipal.SetTextoLog("Imagem cortada para dimensões: ("
                    + FACE_LARGURA + ", " + FACE_ALTURA + ")\n");
        } catch (RasterFormatException e) {
            telaPrincipal.SetTextoLog("Não foi possível cortar a imagem!\n");
            imagemFace = IplImageToBufferedImage(imagemCinza);
        }
        return imagemFace;
    }  // end of CortarFace()

    private void SalvarImagem(BufferedImage im, String fnm) // save image in fnm
    {
        try {
            ImageIO.write(im, "jpg", new File(fnm));
            telaPrincipal.SetTextoLog("Imagem salva em: " + fnm + "\n");
        } catch (IOException e) {
            telaPrincipal.SetTextoLog("Não foi possível salvar imagem: " + fnm + "\n");
        }
    } // end of SalvarImagem()


    /*  Formata a imagem e converte para escala de cinza. FormataÃ§Ã£o deixa a imagem menor
        tornando o processamento mais rÃ¡pido.
        O detector Haar precisa como parÃ¢metro a imagem em escala de cinza    
     */
    private IplImage escalaCinza(IplImage imagemColorida) {
        // Converte para escalas de cinza
        imagemCinza = cvCreateImage(cvGetSize(imagemColorida), IPL_DEPTH_8U, 1);
        cvCvtColor(imagemColorida, imagemCinza, CV_BGR2GRAY);

        // Formata a imagem
        IplImage imagemPequena = IplImage.create(imagemCinza.width() / ESCALA_IMAGEM,
                imagemCinza.height() / ESCALA_IMAGEM, IPL_DEPTH_8U, 1);
        cvResize(imagemCinza, imagemPequena, CV_INTER_LINEAR);

        // Equaliza a imagem menor em escalas de cinza
        cvEqualizeHist(imagemPequena, imagemPequena);
        return imagemPequena;
    }
    

    public static BufferedImage IplImageToBufferedImage(IplImage src) {
        OpenCVFrameConverter.ToIplImage capturaConverter = new OpenCVFrameConverter.ToIplImage();
        Java2DFrameConverter paintConverter = new Java2DFrameConverter();
        org.bytedeco.javacv.Frame frame = capturaConverter.convert(src);
        return paintConverter.getBufferedImage(frame, 1);
    }

} // Fim da classe PainelCadastro


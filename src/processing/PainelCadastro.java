package processing;

/* 
    Este painel repetidamente tira imagens do dispositivo de captura e as desenha no painel.
    Uma face Ã© destacada com um retÃ¢ngulo verde, que Ã© atualizado conforme a face se movimenta.
    
    A face destacada pode ser armazenada para reconhecimento da pessoa.

    A tarefa de detecÃ§Ã£o Ã© realizado pelo Haar cascade classifier disponibilizado pelo JavaCV.
    Esta tarefa Ã© executada em sua prÃ³pria thread devido o processo ser mais lento. sendo assim
    a captura das imagens nÃ£o Ã© afetada pelo reconhecimento dos rostos.
 */
import gui.TelaLog;
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
import org.bytedeco.javacpp.videoInputLib.videoInput;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter.ToIplImage;

public class PainelCadastro extends JPanel implements Runnable {

    private TelaLog log;

    /* DimensÃµes de cada imagem; o painel Ã© do mesmo tamanho das imagens */
    private static final int WIDTH = 640;
    private static final int HEIGHT = 480;

    private static final int DELAY = 100;  // Tempo (ms) delay para desenhar o painel

    private static final int CAMERA_ID = 0;

    private static final int IM_SCALE = 4;
    private static final int SMALL_MOVE = 5;

    // tempo (ms) entre cada detecÃ§Ã£o de face
    private static final int DETECT_DELAY = 500;

    private static final int MAX_TASKS = 4;
    // Nr. mÃ¡ximo de tarefas que podem ficar aguardando no executor

    // Modelo cascata utilizado para detecÃ§Ã£o das faces (Disponibilizado pelo OpenCV)
    private static final String FACE_CASCADE = "haarcascade_frontalface_alt.xml";
    // "haarcascade_frontalface_alt2.xml";

    // Atributos para salvar uma imagem
    private static final String FACE_DIR = "src//fotos";
    private static String FACE_ID;
    private static final int FACE_WIDTH = 125;
    private static final int FACE_HEIGHT = 150;

    private IplImage snapIm = null;
    private volatile boolean isRunning;
    private volatile boolean isFinished;

    // Usado para disponibilizar tempo da captura das imagens
    private int imageCount = 0;
    private long totalTime = 0;
    private Font msgFont;

    // Variáveis do JavaCV
    private FrameGrabber grabber = null;
    private CvHaarClassifierCascade classifier;
    private CvMemStorage storage;
    private CanvasFrame debugCanvas;
    private IplImage imagemCinza;

    // Usado para as threads que executam a detecÃ§Ã£o das faces
    private ExecutorService executor;
    private AtomicInteger numTasks;

    // Usado para armazenar nÃºmero de tarefas de detecÃ§Ã£o
    private long detectStartTime = 0;

    private Rectangle faceRect;             // Armazena as coordenadas da face

    private static volatile boolean salvarFace = false;

    private int amostra = 1;

    public PainelCadastro() {
        setBackground(Color.white);
        msgFont = new Font("SansSerif", Font.BOLD, 18);

        executor = Executors.newSingleThreadExecutor();
        /* O executor controla uma Ãºnica thread com uma fila.
            Somente uma tarefa pode executar de cada vez. As outras devem esperar.
         */
        numTasks = new AtomicInteger(0);
        // Usado para limitar o tamanho da fila do executor.

        initDetector();
        faceRect = new Rectangle();

        isRunning = true;
        isFinished = false;

        log = new TelaLog();
        log.setVisible(true);

        new Thread(this).start();   // Atualiza as imagens para o painel.
    }

    private void initDetector() {
        // Cria uma instÃ¢ncia do cascade classifier para detecÃ§Ã£o dos rostos
        classifier = new CvHaarClassifierCascade(cvLoad(FACE_CASCADE));
        if (classifier.isNull()) {
            System.out.println("\nCould not load the classifier file: " + FACE_CASCADE);
            System.exit(1);
        }

        storage = CvMemStorage.create();  //Cria armazenamento usado para detecÃ§Ã£o

        // debugCanvas = new CanvasFrame("Debugging Canvas");
        // useful for showing JavaCV IplImage objects, to check on image processing
    }  // end of initDetector()

    /*
        Deixa o painel no tamanho suficiente para uma imagem
     */
    public Dimension getPreferredSize() {
        return new Dimension(WIDTH, HEIGHT);
    }

    /*
        Exibe a imagem do dispositivo de captura a cada tempo de DELAY.
        A tarefa de detecÃ§Ã£o somente se inicia apÃ³s cada tempo de DETECT_DELAY
        e somente se o nÃºmero de tarefas no executor for menor que seu limite
     */
    public void run() {
        grabber = initGrabber(CAMERA_ID);
        if (grabber == null) {
            return;
        }

        long duration;

        while (isRunning) {
            long startTime = System.currentTimeMillis();

            snapIm = picGrab(grabber, CAMERA_ID);

            if (((System.currentTimeMillis() - detectStartTime) > DETECT_DELAY)
                    && (numTasks.get() < MAX_TASKS)) {
                trackFace(snapIm);
            }
            imageCount++;
            repaint();
            super.repaint();

            duration = System.currentTimeMillis() - startTime;
            totalTime += duration;

            if (duration < DELAY) {
                try {
                    Thread.sleep(DELAY - duration);  // Aguarda terminar o tempo de delay
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }

        }
        closeGrabber(CAMERA_ID);
        log.setjTextArea1("Fim da execução");
        isFinished = true;

    }

    /*
        Inicia objeto de captura
     */
    private FrameGrabber initGrabber(int ID) {
        log.setjTextArea1("Inicializando captura pelo dispositivo: " + videoInput.getDeviceName(ID) + "\n");
        //System.out.println("Inicializando captura pelo dispositivo: " + videoInput.getDeviceName(ID));
        try {
            if (grabber == null) {
                grabber = FrameGrabber.createDefault(ID);
                grabber.setFormat("dshow");       // Usando DirectShow
                grabber.setImageWidth(WIDTH);     // tamanho padrÃ£o das imagens Ã© pequeno: 320x240
                grabber.setImageHeight(HEIGHT);
            }
            grabber.start();
        } catch (Exception e) {
            System.out.println("Não foi possível carregar a câmera!");
            System.out.println(e);
            System.exit(1);
        }
        return grabber;
    }

    private IplImage picGrab(FrameGrabber grabber, int ID) {
        IplImage im = null;
        OpenCVFrameConverter.ToIplImage conversor = new OpenCVFrameConverter.ToIplImage();
        try {
            im = conversor.convert(grabber.grab());  //Tira uma foto
        } catch (Exception e) {
            System.out.println("Problema ao carregar imagem da câmera " + ID);
        }
        return im;
    }

    /*
        Finaliza a obtenção das imagens do dispositivo de cÃ¢mera
     */
    public void closeGrabber(int ID) {
        try {
            grabber.stop();
            grabber.release();
        } catch (Exception e) {
            System.out.println("Problema ao desativar captura da câmera " + ID);
        }
    }

    /*
        Desenha a imagem, o retângulo em volta a da face detectada e a média de tempo
        de obtenção das imagens da cÃ¢mera no canto inferior esquerdo do painel.
        No canto superior exibe a face identificada.
        O tempo exibido não inclui a tarefa de detecçãoo do rosto.
     */
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setFont(msgFont);

        // Desenha a imagem, estatÃ­sticas e retÃ¢ngulo
        if (snapIm != null) {
            g2.setColor(Color.YELLOW);
            OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
            Java2DFrameConverter paintConverter = new Java2DFrameConverter();
            org.bytedeco.javacv.Frame frame = grabberConverter.convert(snapIm);
            g2.drawImage(paintConverter.getBufferedImage(frame, 1), 0, 0, this);   // Desenha a imagem
            String statsMsg = String.format("Tempo médio de captura:  %.1f ms",
                    ((double) totalTime / imageCount));

            // Escreve as estatÃ­sticas no canto inferior esquerdo
            g2.drawString(statsMsg, 5, HEIGHT - 10);

            drawRect(g2);
        } else {  // Caso ainda nÃ£o obtiver nenhuma imagem
            g2.setColor(Color.BLUE);
            g2.drawString("Carregando câmera: " + CAMERA_ID, 5, HEIGHT - 10);
        }
    }

    /*
        Usa o retÃ¢ngulo da face a desenhar o retÃ¢ngulo verde em torno da face
        O desenho do retÃ¢ngulo estÃ¡ em um bloco sincronizado pois a variÃ¡vel faceRect pode estar sendo
        utilizada em outra thread.
     */
    private void drawRect(Graphics2D g2) {
        synchronized (faceRect) {
            if (faceRect.width == 0) {
                return;
            }

            // Desenha o retÃ¢ngulo
            g2.setColor(Color.GREEN);
            g2.setStroke(new BasicStroke(2));
            g2.drawRect(faceRect.x, faceRect.y, faceRect.width, faceRect.height);

        }
    }

    /*  Termina a execuÃ§Ã£o do run() e espera a finalizaÃ§Ã£o do sistema.
        Para a aplicaÃ§Ã£o atÃ© que sua execuÃ§Ã£o termine.
     */
    public void closeDown() {
        isRunning = false;
        log.dispose();
        while (!isFinished) {
            try {
                Thread.sleep(DELAY);
            } catch (Exception ex) {
            }
        }
    }

    // ------------------------- face tracking ----------------------------\\
    /*  Cria um thread para detectar faces nas imagens geradas pela cÃ¢mera.
        Armazena as coordenadas das faces em faceRect e salva a imagem para treinamento caso solicitado.
        Imprime o tempo de execuÃ§Ã£o no console
     */
    private void trackFace(IplImage img) {
        numTasks.getAndIncrement();     // Incrementa o nÃºmero de tarefas antes de entrar na fila
        executor.execute(new Runnable() {
            public void run() {
                detectStartTime = System.currentTimeMillis();
                CvRect rect = findFace(img);
                if (rect != null) {

                    // Seta parÃ¢metros do retÃ¢ngulo
                    setRectangle(rect);
                } else {
                    faceRect.setBounds(0, 0, 0, 0);
                }
                long detectDuration = System.currentTimeMillis() - detectStartTime;
                log.setjTextArea1(" duração da detecção: " + detectDuration + "ms\n");
                //System.out.println(" detection duration: " + detectDuration + "ms");
                numTasks.getAndDecrement();  // Decrementa o nÃºmero de tarefas quando terminado
            }
        });
    }

    /*  Formata a imagem e converte para escala de cinza. FormataÃ§Ã£o deixa a imagem menor
        tornando o processamento mais rÃ¡pido.
        O detector Haar precisa como parÃ¢metro a imagem em escala de cinza    
     */
 /*  Detecta somente uma face utilizando o Haar Detector
        Chama a funÃ§Ã£o para a identificaÃ§Ã£o do rosto encontrado.
     */
    private CvRect findFace(IplImage imagemColorida) {

        // Converte para escalas de Cinza
        imagemCinza = escalaCinza(imagemColorida);

        /*
     // Mostra a imagem em escalas de cinza para verificar o processo de tratamento da imagem
     debugCanvas.showImage(grayIm);
	 debugCanvas.waitKey(0);
         */
        // System.out.println("Detecting largest face...");   // cvImage
        CvSeq faces = cvHaarDetectObjects(imagemCinza, classifier, storage, 1.1, 1, // 3
                // CV_HAAR_SCALE_IMAGE |
                CV_HAAR_DO_ROUGH_SEARCH | CV_HAAR_FIND_BIGGEST_OBJECT);
        // Por questÃµes de seguranÃ§a e rapidez, sÃ³ serÃ¡ detectado o rosto maior e mais prÃ³ximo.

        int total = faces.total();
        if (total == 0) {
            log.setjTextArea1("Face não encontrada\n");
            //System.out.println("No faces found");
            return null;
        } else if (total > 1) //Este caso não deveria ocorrer. IncluÃ­do por seguranÃ§a
        {
            log.setjTextArea1("Multiplas faces detectadas (" + total + "); utilizando a primeira\n");
            // System.out.println("Multiple faces detected (" + total + "); using the first");
        } else {
            log.setjTextArea1("Face detectada\n");
            //System.out.println("Face detected");
        }

        CvRect rect = new CvRect(cvGetSeqElem(faces, 0));

        if (salvarFace) {
            clipSaveFace(imagemColorida);
        }

        cvClearMemStorage(storage);
        return rect;
    }

    /* Extrai o tamanho e as coordenadas da imagem desatacada da estrutura do retÃ¢ngulo do JavaCV
       e armazena em um retÃ¢ngulo Java.
       Durante o processo, desfaz o escalonamento que foi aplicado a imagem antes da detecÃ§Ã£o do rosto.
       Disponibiliza informaÃ§Ãµes de movimento do rosto na imagem.
       O uso dessa funÃ§Ã£o estÃ¡ em um bloco sincronizado pois o retÃ¢ngulo pode estar sendo utilizado para
       atualizar o painel de imagem ou pintura do retÃ¢ngulo ao mesmo tempo em outras threads.
     */
    private void setRectangle(CvRect r) {
        synchronized (faceRect) {
            int xNew = r.x() * IM_SCALE;
            int yNew = r.y() * IM_SCALE;
            int widthNew = r.width() * IM_SCALE;
            int heightNew = r.height() * IM_SCALE;

            // calcula o movimento do retÃ¢ngulo comparado com o anterior
            int xMove = (xNew + widthNew / 2) - (faceRect.x + faceRect.width / 2);
            int yMove = (yNew + heightNew / 2) - (faceRect.y + faceRect.height / 2);

            // DispÃµe informaÃ§Ãµes de movimento da face se for significante
            if ((Math.abs(xMove) > SMALL_MOVE) || (Math.abs(yMove) > SMALL_MOVE)) {
                log.setjTextArea1("Movimento (x,y): (" + xMove + "," + yMove + ")\n");
                //System.out.println("Movement (x,y): (" + xMove + "," + yMove + ")");
            }

            faceRect.setRect(xNew, yNew, widthNew, heightNew);
        }
    }

    // --------------------------- Salvar e Aprender nova Face -----------------------------------
    public static void saveFace(String faceID) {
        salvarFace = true;
        FACE_ID = faceID;
    }

    private void clipSaveFace(IplImage img) /* clip the image using the current face rectangle, and save it into fnm
     The use of faceRect is in a synchronized block since it may be being
     updated or used for drawing at the same time in other threads.
     */ {
        BufferedImage clipIm = null;
        synchronized (faceRect) {
            if (faceRect.width == 0) {
                System.out.println("No face selected");
                return;
            }
            BufferedImage im = IplImageToBufferedImage(img);
            try {
                clipIm = im.getSubimage(faceRect.x, faceRect.y, faceRect.width, faceRect.height);
            } catch (RasterFormatException e) {
                System.out.println("Could not clip the image");
            }
        }
        if (clipIm != null) {
            saveClip(clipIm);
        }
    }  // end of clipSaveFace()

    private void saveClip(BufferedImage clipIm) /* resizes to at least a standard size, converts to grayscale, 
     clips to an exact size, then saves in a standard location */ {
        long startTime = System.currentTimeMillis();

        System.out.println("Saving clip...");
        BufferedImage grayIm = resizeImage(clipIm);
        BufferedImage faceIm = clipToFace(grayIm);
        saveImage(faceIm, FACE_DIR + "/" + FACE_ID + "-" + amostra + ".jpg");
        amostra++;
        salvarFace = false;
        System.out.println("  Save time: " + (System.currentTimeMillis() - startTime) + " ms");
    }  // end of saveClip()

    private BufferedImage resizeImage(BufferedImage im) // resize to at least a standard size, then convert to grayscale 
    {
        // resize the image so *at least* FACE_WIDTH*FACE_HEIGHT size
        int imWidth = im.getWidth();
        int imHeight = im.getHeight();
        System.out.println("Original (w,h): (" + imWidth + ", " + imHeight + ")");

        double widthScale = FACE_WIDTH / ((double) imWidth);
        double heightScale = FACE_HEIGHT / ((double) imHeight);
        double scale = (widthScale > heightScale) ? widthScale : heightScale;

        int nWidth = (int) Math.round(imWidth * scale);
        int nHeight = (int) Math.round(imHeight * scale);

        // convert to grayscale while resizing
        BufferedImage grayIm = new BufferedImage(nWidth, nHeight,
                BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2 = grayIm.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2.drawImage(im, 0, 0, nWidth, nHeight, 0, 0, imWidth, imHeight, null);
        g2.dispose();

        System.out.println("Scaled gray (w,h): (" + nWidth + ", " + nHeight + ")");
        return grayIm;
    }  // end of resizeImage()

    private BufferedImage clipToFace(BufferedImage im) // clip image to FACE_WIDTH*FACE_HEIGHT size
    // I assume the input image is face size or bigger
    {
        int xOffset = (im.getWidth() - FACE_WIDTH) / 2;
        int yOffset = (im.getHeight() - FACE_HEIGHT) / 2;
        BufferedImage faceIm = null;
        try {
            faceIm = im.getSubimage(xOffset, yOffset, FACE_WIDTH, FACE_HEIGHT);
            System.out.println("Clipped image to face dimensions: ("
                    + FACE_WIDTH + ", " + FACE_HEIGHT + ")");
        } catch (RasterFormatException e) {
            System.out.println("Could not clip the image");
            faceIm = IplImageToBufferedImage(imagemCinza);
        }
        return faceIm;
    }  // end of clipToFace()

    private void saveImage(BufferedImage im, String fnm) // save image in fnm
    {
        try {
            ImageIO.write(im, "jpg", new File(fnm));
            System.out.println("Saved image to " + fnm);
        } catch (IOException e) {
            System.out.println("Could not save image to " + fnm);
        }
    }  // end of saveImage()

    /*  Formata a imagem e converte para escala de cinza. FormataÃ§Ã£o deixa a imagem menor
        tornando o processamento mais rÃ¡pido.
        O detector Haar precisa como parÃ¢metro a imagem em escala de cinza    
     */
    private IplImage escalaCinza(IplImage imagemColorida) {
        // Converte para escalas de cinza
        IplImage imagemCinza = cvCreateImage(cvGetSize(imagemColorida), IPL_DEPTH_8U, 1);
        cvCvtColor(imagemColorida, imagemCinza, CV_BGR2GRAY);

        // Formata a imagem
        IplImage imagemPequena = IplImage.create(imagemCinza.width() / IM_SCALE,
                imagemCinza.height() / IM_SCALE, IPL_DEPTH_8U, 1);
        cvResize(imagemCinza, imagemPequena, CV_INTER_LINEAR);

        // Equaliza a imagem menor em escalas de cinza
        cvEqualizeHist(imagemPequena, imagemPequena);
        return imagemPequena;
    }
    
    /* Converte uma BufferedImage para IplImage para utilizaÃ§Ã£o do OpenCV */
    private IplImage toIplImage(BufferedImage bufImage) {
        ToIplImage iplConverter = new OpenCVFrameConverter.ToIplImage();
        Java2DFrameConverter java2dConverter = new Java2DFrameConverter();
        IplImage iplImage = iplConverter.convert(java2dConverter.convert(bufImage));
        return iplImage;
    }

    public static BufferedImage IplImageToBufferedImage(IplImage src) {
        OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
        Java2DFrameConverter paintConverter = new Java2DFrameConverter();
        org.bytedeco.javacv.Frame frame = grabberConverter.convert(src);
        return paintConverter.getBufferedImage(frame, 1);
    }

} // end of PainelCadastro class


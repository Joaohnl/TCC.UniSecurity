package gui;


/* Mostra uma sequência de imagens capturadas da webcam em um Painel (FacePanel).
   Uma face é destacada com um quadrado verde.
   É possível salvar a imagem destacada apertado o botão salvar face.

 */
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_objdetect;
import processing.FacePanel;

public class telaPrincipal extends JFrame {
    // Componentes da interface do usuário

    private FacePanel facePanel;
    private JButton btnTreinarModelo;

    public telaPrincipal() {
        super("UniSecurity");

        Container c = getContentPane();
        c.setLayout(new BorderLayout());

        // précarrega módulo opencv_objdetect para resolver bug conhecido.
        Loader.load(opencv_objdetect.class);

        facePanel = new FacePanel(); // sequência de imagens aparece aqui.
        c.add(facePanel, BorderLayout.CENTER);

        // Botão para salvar face detectada
        JButton btnSalvarFace = new JButton("Salvar Face");
        btnTreinarModelo = new JButton("Treinar Modelo");

        JLabel lbl_RA = new JLabel("Insira o RA:");

        JTextField txtIdentificacao = new JTextField("Insira o RA do aluno");

//        txtIdentificacao.addKeyListener(new KeyAdapter() {
//            // Habilita ou desabilita o botão de salvar face.
//            // Campo do RA precisa ter 9 digítos (tamanho do RA dos alunos) para habilitar o botão.
//            @Override
//            public void keyReleased(KeyEvent e) { // Observa entrada de dígitos no campo
//                try {
//                    int teste = Integer.parseInt(txtIdentificacao.getText());
//                    if (txtIdentificacao.getText().length() != 9) {
//                        btnSalvarFace.setEnabled(false);
//                    } else {
//                        btnSalvarFace.setEnabled(true);
//                    }
//                } catch (NumberFormatException exc) {
//                    txtIdentificacao.setText("");
//                }
//
//            }
//        });
        btnSalvarFace.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                facePanel.saveFace(txtIdentificacao.getText());
            }
        });

        //btnSalvarFace.setEnabled(false);
        JPanel p = new JPanel();
        p.add(lbl_RA);
        p.add(txtIdentificacao);
        p.add(btnSalvarFace);
        p.add(btnTreinarModelo);
        c.add(p, BorderLayout.SOUTH);

        addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                facePanel.closeDown();    // para de tirar imagens do dispositivo de câmera
                System.exit(0);
            }
        });

        setResizable(false);
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }
} //Fim da classe telaPrincipal

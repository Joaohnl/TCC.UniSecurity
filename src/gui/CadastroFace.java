package gui;


/* Mostra uma sequência de imagens capturadas da webcam em um Painel (PainelCadastro).
   Uma face é destacada com um quadrado verde.
   É possível salvar a imagem destacada apertado o botão salvar face.

 */
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import processing.PainelCadastro;

public class CadastroFace extends JFrame {
    // Componentes da interface do usuário

    private PainelCadastro facePanel;

    public CadastroFace() {
        super("Preview");

        Container c = getContentPane();
        c.setLayout(new BorderLayout());

        facePanel = new PainelCadastro(); // sequência de imagens aparece aqui.
        c.add(facePanel, BorderLayout.CENTER);

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


        setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);

        setResizable(false);
        pack();
        setLocation(400, 100);
        setVisible(true);
    }

    public PainelCadastro getFacePanel() {
        return facePanel;
    }

    public void setFacePanel(PainelCadastro facePanel) {
        this.facePanel = facePanel;
    }
    
    
} //Fim da classe telaPrincipal

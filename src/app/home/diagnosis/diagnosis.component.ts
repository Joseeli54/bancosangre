import { Component, ElementRef, Input, OnInit, ViewChild } from '@angular/core';
import { Table } from 'primeng/table';
import { Diagnostico } from 'src/app/models/diagnostico';
import { Sangre } from 'src/app/models/sangre';
import { UserServices } from '../../services/userservices.services';
import { NgbModal, ModalDismissReasons } from '@ng-bootstrap/ng-bootstrap';
import { AlertService } from '../../services/alert/alert.service';
import { environment } from '../../../environments/environment';
import { User } from 'src/app/models/user';


@Component({
  selector: 'app-diagnosis',
  templateUrl: './diagnosis.component.html',
  styleUrls: ['./diagnosis.component.css']
})
export class DiagnosisComponent {
  @Input() content_diagnostico : Array<Diagnostico> = [];
  
  @ViewChild('dt') dt? : Table;
  
  //Variables que serán utilizadas dentro de la tabla
  public _selected_columns: any = [];
  public lazy = false;
  public show_filters = true;

  //Objeto para guardar la información del sangreo seleccionado
  public diagnostico : Diagnostico = {};
  public sangres : Array<Sangre> = [];
  
  //Objetos que serán utilizado para llamar el modal
  public modalReference : any;
  public closeResult: string | undefined;
  public titleModal : string = "";
  public usuarios : Array<User> = [];

  public table_cols = [
    { field: 'name', header: 'Nombre' },
    { field: 'description', header: 'Observación' },
    { field: 'id_sangre', header: 'Sangre Asociada' },
    { field: 'id_user', header: 'Donante' },
    { field: 'createBy', header: 'Creado Por' },
    { field: 'actions', header: 'Acciones' },
  ];

  public action = "";

  constructor(public userService: UserServices,
              private modalService: NgbModal,
              private alertService: AlertService) { 
    localStorage.setItem('view', '3');
  }

  ngOnInit(): void {
    this._selected_columns = this.table_cols;
    this.getDiagnosticos();
    this.getUsers();
    this.getSangres();
  }

  public getDiagnosticos(){
    localStorage.setItem('spinner', '1');

    this.userService
      .getUrl('diagnosticos')
      .then(response => {
        this.content_diagnostico = response;
        console.log(this.content_diagnostico)
        localStorage.setItem('spinner', '0');
      })
      .catch(error => {
        console.log(error.error)
        this.alertService.errorOcurred(error.statusText);
      });
  }

  public getSangres(){
    localStorage.setItem('spinner', '1');

    this.userService
      .getUrl('sangres')
      .then(response => {
        this.sangres = response;
        console.log(this.sangres)
        localStorage.setItem('spinner', '0');
      })
      .catch(error => {
        console.log(error.error)
        this.alertService.errorOcurred(error.statusText);
      });
  }

  public getUsers(){
    this.userService.getUrl('users')
    .then(data => {

      this.usuarios = [];
      for (let index = 0; index < data.length; index++) {
        if(data[index].role == "USER"){
          this.usuarios.push(data[index]);
        }
      }
    })
    .catch(err => {
      console.log(err);
    })
  }

  public clearData(){
    this.diagnostico = new Sangre();
  }

  public deleteDiagnostico(id : any){
    this.userService
    .deleteUrl('diagnosticos/{id}', [id])
    .then(response => {
      console.log(response);
      this.getDiagnosticos();
      this.modalReference.close();
    })
    .catch(error => {
      console.log(error);
    })
  }

  open(content: any, data : any) {

    if(data != null){
      this.diagnostico._id = data._id;
      this.diagnostico.name = data.name;
      this.diagnostico.description = data.description;
      this.diagnostico.id_sangre = data.id_sangre;
      this.diagnostico.id_user = data.id_user;
      this.diagnostico.createBy = data.createBy;
      this.titleModal = "Editar Diagnostico";

      this.action = "U";
    }else{
      this.diagnostico.name = "Diagnostico #" + (this.content_diagnostico.length + 1);
      this.diagnostico.description = "";
      this.diagnostico.id_sangre = "1";
      this.diagnostico.id_user = "1";
      this.diagnostico.createBy = localStorage.getItem('username')!;
      this.titleModal = "Agregar Diagnostico";
      this.action = "C";
    }

    this.modalReference = this.modalService.open(content, { size: 'lg', centered: true});

    this.modalReference.result.then((result: any) => {
      this.closeResult = `Closed with: ${result}`;
    }, (reason: any) => {
      this.clearData();
      this.closeResult = `Dismissed ${this.getDismissReason(reason)}`;
    });
  }

  private getDismissReason(reason: any): string {
    if (reason === ModalDismissReasons.ESC) {
      return 'by pressing ESC';
    } else if (reason === ModalDismissReasons.BACKDROP_CLICK) {
      return 'by clicking on a backdrop';
    } else {
      return  `with: ${reason}`;
    }
  }

  public createOrUpdateSangre(){
    console.log(this.diagnostico);

    let diagnosticos = {
      'name': this.diagnostico.name,
      'description': this.diagnostico.description,
      'id_sangre': this.diagnostico.id_sangre,
      'id_user': this.diagnostico.id_user,
      'createBy': this.diagnostico.createBy }

    localStorage.setItem('spinner', '1');
    if(this.action == "C"){
      this.userService
        .postUrl('diagnosticos', diagnosticos)
        .then(response => {
          console.log(response)
          this.getDiagnosticos();
          this.modalReference.close();
        })
    }else{
      let id : any = this.diagnostico._id;
      this.userService
        .putUrl('diagnosticos/{id}', diagnosticos, [id])
        .then(response => {
          console.log(response);
          this.getDiagnosticos();
          this.modalReference.close();
        })
        .catch(error => {
          console.log(error);
        })
    }
  }

  public userName(_id : any){
    let name = "No Aplicado";

    for (let index = 0; index < this.usuarios.length; index++) {
      if(this.usuarios[index]._id == _id){
        name = this.usuarios[index].name + " " + this.usuarios[index].apellido;
      }
    }

    return name;
  }

  delete(content: any, data : any){
    if(data != null){
      this.diagnostico._id = data._id;
    }

    this.modalReference = this.modalService.open(content, { centered: true});

    this.modalReference.result.then((result: any) => {
      this.closeResult = `Closed with: ${result}`;
    }, (reason: any) => {
      this.clearData();
      this.closeResult = `Dismissed ${this.getDismissReason(reason)}`;
    });
  }

  @Input() get selected_columns(): any[] {
    return this._selected_columns;
  }

  set selected_columns(val: any[]) {
    //restore original order
    this._selected_columns = this.table_cols.filter(col => val.includes(col));
  }

  //PrimeNG Functions and Procedures
  apply_filter($event : any) {
    console.log($event.target.value);
    this.dt?.filterGlobal(($event.target as HTMLInputElement).value, 'contains');
  }

  clear(table: Table) {
    table.clear();
  }
}

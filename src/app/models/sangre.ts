import {Entity} from '../classes/entity'

export class Sangre extends Entity{
    constructor(
        public code?:number,
        public name?:string,
        public content?:string,
        public level?:string,
        public type?:string,
        public grupo?:string,
        public factor_rh?:string,
        public date_donor?:Date,
        public date_due?:Date,
        public id_user?:string,
        public createBy?:any,
    ){
        super();
    }
}